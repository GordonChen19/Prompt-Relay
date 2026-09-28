"""Independent, overlapping schedules on H3's actual (non-uniform) time grid."""

import math
from bisect import bisect_left
from dataclasses import dataclass

import torch


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_prompt_text(config):
    """Return the complete prompt and character spans in local-prompt order."""
    if not isinstance(config, dict):
        raise ValueError("prompt_relay_config must be a JSON object.")
    global_prompt = config.get("global_prompt", "")
    local = config.get("local_prompts", [])
    if not isinstance(global_prompt, str):
        raise ValueError("global_prompt must be a string.")
    if not isinstance(local, list) or any(not isinstance(p, str) or not p.strip() for p in local):
        raise ValueError("local_prompts must be a list of nonempty strings.")
    text, spans = global_prompt.strip(), []
    for prompt in local:
        text += "\n" if text else ""
        start = len(text)
        text += prompt.strip()
        spans.append((start, len(text)))
    if not text:
        raise ValueError("The combined prompt must not be empty.")
    return text, spans


def token_segment_ids(tokenizer, config, num_text_tokens):
    """Map offsets from ONE full tokenization; media-prefix tokens stay global.

    All three pinned upstream H3 encoders append the separately tokenized prompt
    verbatim after their media presentation, without a chat template or special
    tokens. Repeated local strings therefore remain unambiguous.
    """
    text, spans = build_prompt_text(config)
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(encoded["input_ids"]) or num_text_tokens < len(offsets):
        raise ValueError("H3 prompt embeddings do not match the complete, untruncated prompt.")
    prefix_length = num_text_tokens - len(offsets)
    ids = torch.full((num_text_tokens,), -1, dtype=torch.long)
    for token_index, (start, end) in enumerate(offsets):
        matches = [i for i, (a, b) in enumerate(spans) if start < b and end > a]
        if len(matches) > 1:
            raise ValueError("A tokenizer token crosses two local prompts; revise the prompt boundary.")
        if matches:
            a, b = spans[matches[0]]
            outside = text[start:min(end, a)] + text[max(start, b):end]
            if outside.strip():
                raise ValueError("A tokenizer token crosses global/local content; revise the prompt boundary.")
            ids[prefix_length + token_index] = matches[0]
    for index in range(len(spans)):
        if not torch.any(ids == index):
            raise ValueError(f"local_prompts[{index}] has no encoded tokens.")
    return ids


@dataclass(frozen=True)
class Schedule:
    intervals: list
    coordinates: torch.Tensor
    midpoints: torch.Tensor
    half_widths: torch.Tensor
    sigma: float


def prepare_schedule(config, frame_times, duration):
    """Quantize half-open intervals onto actual H3 target-video timestamps.

    Seconds use H3's shared 40 Hz rotary clock after removing the target origin.
    Internal-frame intervals address transformer frames, not decoded frames.
    Local intervals are independent: overlaps and uncovered gaps are allowed.
    """
    _, spans = build_prompt_text(config)
    count = len(spans)
    times = torch.as_tensor(frame_times, dtype=torch.float64).cpu()
    if times.ndim != 1 or len(times) == 0 or not torch.isfinite(times).all():
        raise ValueError("frame_times must contain finite target-video timestamps.")
    if times[0] < 0 or torch.any(times[1:] <= times[:-1]) or not _number(duration) or duration <= times[-1]:
        raise ValueError("frame_times must increase within the output duration.")
    total = len(times)
    allowed = {"global_prompt", "local_prompts", "segment_intervals", "segment_lengths", "time_unit",
               "tail_width", "epsilon", "routing_mode"}
    if unknown := config.keys() - allowed:
        raise ValueError(f"Unsupported H3 Prompt Relay fields: {sorted(unknown)}. Use explicit segment_intervals for overlap.")
    if config.get("routing_mode", "overlap") != "overlap":
        raise ValueError("H3 supports routing_mode='overlap' only.")
    explicit = "segment_intervals" in config
    if explicit and "segment_lengths" in config:
        raise ValueError("Use segment_intervals or segment_lengths, not both.")
    unit = config.get("time_unit", "internal_frame")
    if unit not in ("internal_frame", "seconds"):
        raise ValueError("time_unit must be 'internal_frame' or 'seconds'.")
    if not explicit and unit != "internal_frame":
        raise ValueError("time_unit='seconds' requires segment_intervals.")
    epsilon = config.get("epsilon", 1e-3)
    tail = config.get("tail_width", 8 / 24 if unit == "seconds" else 2.0)
    if not _number(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and strictly between zero and one.")
    if not _number(tail) or tail <= 0:
        raise ValueError("tail_width must be finite and positive, in time_unit units.")
    sigma = tail / math.sqrt(-2 * math.log(epsilon))
    if not 1e-15 <= sigma <= 1e15:
        raise ValueError("tail_width produces an unrepresentable float32 decay.")
    intervals = []
    if explicit:
        supplied = config["segment_intervals"]
        if not isinstance(supplied, list) or len(supplied) != count:
            raise ValueError("segment_intervals must contain one [start, end] pair per local prompt.")
        limit = duration if unit == "seconds" else total
        for index, pair in enumerate(supplied):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"segment_intervals[{index}] must be [start, end].")
            a, b = pair
            if not all(_number(v) for v in pair) or not 0 <= a < b <= limit:
                raise ValueError(f"segment_intervals[{index}] must satisfy 0 <= start < end <= {limit}.")
            if unit == "seconds":
                # Float64 RoPE origins may differ by a few ulps after references.
                a, b = bisect_left(times.tolist(), a - 1e-9), bisect_left(times.tolist(), b - 1e-9)
            elif any(not isinstance(v, int) for v in pair):
                raise ValueError("internal_frame endpoints must be integers.")
            if a == b:
                raise ValueError(f"segment_intervals[{index}] contains no target-video timestamp; widen it.")
            intervals.append((a, b))
    elif count:
        lengths = config.get("segment_lengths", [])
        if not isinstance(lengths, list):
            raise ValueError("segment_lengths must be a list.")
        if not lengths:
            base, remainder = divmod(total, count)
            lengths = [base + (i < remainder) for i in range(count)]
        if len(lengths) != count or any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != total:
            raise ValueError(f"segment_lengths must have one positive integer per local prompt and sum to {total}.")
        start = 0
        for length in lengths:
            intervals.append((start, start + length))
            start += length
    elif config.get("segment_lengths"):
        raise ValueError("segment_lengths requires local_prompts.")
    coordinates = times.float() if unit == "seconds" else torch.arange(total, dtype=torch.float32)
    midpoints = torch.tensor([(coordinates[a] + coordinates[b - 1]) / 2 for a, b in intervals])
    half_widths = torch.tensor([(coordinates[b - 1] - coordinates[a]) / 2 for a, b in intervals])
    return Schedule(intervals, coordinates, midpoints, half_widths, sigma)
