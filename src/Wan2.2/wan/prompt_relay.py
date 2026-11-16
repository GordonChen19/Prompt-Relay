"""Configuration and temporal routing helpers for Prompt Relay."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SUPPORTED_RELAY_MODES = frozenset({"gaussian", "hard"})


@dataclass(frozen=True)
class PromptRelayConfig:
    """Validated user configuration loaded from a Prompt Relay JSON file."""

    global_prompt: str
    local_prompts: tuple[str, ...]
    id: str | None = None
    file_name: str | None = None
    segment_lengths: tuple[int, ...] = ()
    explicit_frame_intervals: tuple[tuple[int, int], ...] = ()
    mode: str = "gaussian"
    sigma: float = 0.932

    @classmethod
    def from_json(cls, path: str | Path) -> "PromptRelayConfig":
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise ValueError(f"Prompt Relay file does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Prompt Relay file is not valid JSON: {path}: {exc.msg}"
            ) from exc

        if not isinstance(data, Mapping):
            raise ValueError("Prompt Relay JSON must contain an object at its root")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PromptRelayConfig":
        global_prompt = data.get("global_prompt")
        local_prompts = data.get("local_prompts")
        identifier = data.get("id")
        file_name = data.get("file_name")
        segment_lengths = data.get("segment_lengths", ())
        frame_intervals = data.get("frame_intervals", ())
        mode = data.get("mode", "gaussian")
        sigma = data.get("sigma", 0.932)

        if not isinstance(global_prompt, str) or not global_prompt.strip():
            raise ValueError("global_prompt must be a non-empty string")
        if not isinstance(local_prompts, Sequence) or isinstance(
            local_prompts, (str, bytes)
        ):
            raise ValueError("local_prompts must be a non-empty array of strings")

        normalized_prompts = tuple(
            prompt.strip() if isinstance(prompt, str) else prompt
            for prompt in local_prompts
        )
        if not normalized_prompts or any(
            not isinstance(prompt, str) or not prompt
            for prompt in normalized_prompts
        ):
            raise ValueError("local_prompts must be a non-empty array of strings")

        if identifier is not None:
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("id must be a non-empty string when provided")
            identifier = identifier.strip()

        if file_name is not None:
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError(
                    "file_name must be a non-empty string when provided"
                )
            file_name = file_name.strip()
            if Path(file_name).name != file_name or any(
                separator in file_name for separator in ("/", "\\")
            ):
                raise ValueError(
                    "file_name must be a filename, not a directory path"
                )
            if Path(file_name).suffix.lower() != ".mp4":
                raise ValueError("file_name must end with .mp4")

        if not isinstance(segment_lengths, Sequence) or isinstance(
            segment_lengths, (str, bytes)
        ):
            raise ValueError("segment_lengths must be an array of positive integers")
        normalized_lengths = tuple(segment_lengths)
        if any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            for length in normalized_lengths
        ):
            raise ValueError("segment_lengths must contain only positive integers")
        if normalized_lengths and len(normalized_lengths) != len(
            normalized_prompts
        ):
            raise ValueError(
                "segment_lengths must contain one entry per local prompt"
            )

        if not isinstance(frame_intervals, Sequence) or isinstance(
            frame_intervals, (str, bytes)
        ):
            raise ValueError(
                "frame_intervals must be an array of [start, end] pairs"
            )
        normalized_intervals = []
        for interval in frame_intervals:
            if (
                not isinstance(interval, Sequence)
                or isinstance(interval, (str, bytes))
                or len(interval) != 2
            ):
                raise ValueError(
                    "frame_intervals must contain only [start, end] pairs"
                )
            start, end = interval
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                raise ValueError(
                    "Each frame interval must contain integers satisfying "
                    "0 <= start < end"
                )
            normalized_intervals.append((start, end))
        normalized_intervals = tuple(normalized_intervals)

        if normalized_lengths and normalized_intervals:
            raise ValueError(
                "segment_lengths and frame_intervals are mutually exclusive"
            )
        if normalized_intervals and len(normalized_intervals) != len(
            normalized_prompts
        ):
            raise ValueError(
                "frame_intervals must contain one entry per local prompt"
            )

        if not isinstance(mode, str) or mode.lower() not in SUPPORTED_RELAY_MODES:
            supported = ", ".join(sorted(SUPPORTED_RELAY_MODES))
            raise ValueError(f"mode must be one of: {supported}")
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
            raise ValueError("sigma must be a positive number")
        if not math.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError("sigma must be a positive finite number")

        return cls(
            global_prompt=global_prompt.strip(),
            local_prompts=normalized_prompts,
            id=identifier,
            file_name=file_name,
            segment_lengths=normalized_lengths,
            explicit_frame_intervals=normalized_intervals,
            mode=mode.lower(),
            sigma=float(sigma),
        )

    @property
    def prefixed_local_prompts(self) -> tuple[str, ...]:
        """Return local prompts with the leading space used during tokenization."""

        return tuple(f" {prompt}" for prompt in self.local_prompts)

    @property
    def full_prompt(self) -> str:
        return self.global_prompt + "".join(self.prefixed_local_prompts)

    def frame_intervals(self, latent_frames: int) -> tuple[tuple[int, int], ...]:
        """Resolve local prompts to half-open latent-frame intervals."""

        if latent_frames <= 0:
            raise ValueError("latent_frames must be positive")

        if self.explicit_frame_intervals:
            if any(
                end > latent_frames
                for _, end in self.explicit_frame_intervals
            ):
                raise ValueError(
                    "frame_intervals cannot extend beyond the number of latent "
                    f"frames ({latent_frames})"
                )
            return self.explicit_frame_intervals

        if self.segment_lengths:
            if sum(self.segment_lengths) != latent_frames:
                raise ValueError(
                    "segment_lengths must sum to the number of latent frames "
                    f"({latent_frames}), but sum to {sum(self.segment_lengths)}"
                )
            boundaries = [0]
            for length in self.segment_lengths:
                boundaries.append(boundaries[-1] + length)
            return tuple(zip(boundaries, boundaries[1:]))

        if len(self.local_prompts) > latent_frames:
            raise ValueError(
                "There cannot be more local prompts than latent frames "
                f"({len(self.local_prompts)} prompts, {latent_frames} frames)"
            )

        step = math.ceil(latent_frames / len(self.local_prompts))
        return tuple(
            (step * index, min(step * (index + 1), latent_frames))
            for index in range(len(self.local_prompts))
        )


@dataclass(frozen=True)
class PromptRelaySegment:
    """Model-ready routing information for one local prompt."""

    frame_start: int
    frame_end: int
    token_indices: tuple[int, ...]
    tokens_per_frame: int
    mode: str
    sigma: float

    @property
    def midpoint(self) -> int:
        return (self.frame_start + self.frame_end) // 2

    @property
    def inner_window(self) -> int:
        return (self.frame_end - self.frame_start) // 2 - 2


def build_prompt_relay_segments(
    tokenizer,
    relay_config: PromptRelayConfig,
    *,
    latent_frames: int,
    tokens_per_frame: int,
    text_len: int,
) -> tuple[tuple[PromptRelaySegment, ...], str]:
    """Tokenize a relay prompt and map local token spans to video intervals."""

    full_ids = tokenizer(
        relay_config.full_prompt,
        add_special_tokens=True,
        padding=False,
        return_mask=False,
    )[0].tolist()
    if len(full_ids) > text_len:
        raise ValueError(
            "Prompt Relay text uses "
            f"{len(full_ids)} tokens, exceeding the model limit of {text_len}"
        )

    token_spans = []
    search_start = 0
    for prompt in relay_config.prefixed_local_prompts:
        prompt_ids = tokenizer(
            prompt,
            padding=False,
            add_special_tokens=False,
        )[0].tolist()
        match = _find_token_subsequence(
            full_ids,
            prompt_ids,
            start=search_start,
        )
        if match is None:
            raise ValueError(
                "A local prompt could not be located in the tokenized full "
                f"prompt: {prompt.strip()}"
            )
        token_spans.append(tuple(range(*match)))
        search_start = match[1]

    intervals = relay_config.frame_intervals(latent_frames)
    segments = tuple(
        PromptRelaySegment(
            frame_start=frame_start,
            frame_end=frame_end,
            token_indices=token_indices,
            tokens_per_frame=tokens_per_frame,
            mode=relay_config.mode,
            sigma=relay_config.sigma,
        )
        for (frame_start, frame_end), token_indices in zip(
            intervals,
            token_spans,
        )
    )
    return segments, relay_config.full_prompt


def _find_token_subsequence(haystack, needle, start=0):
    if not needle:
        return None
    last_start = len(haystack) - len(needle)
    for index in range(start, last_start + 1):
        if haystack[index:index + len(needle)] == needle:
            return index, index + len(needle)
    return None


def build_temporal_penalty(
    segments: Sequence[PromptRelaySegment],
    query_length: int,
    context_length: int,
    *,
    device: torch.device,
    query_token_offset: int = 0,
) -> torch.Tensor:
    """Build the float32 penalty subtracted from cross-attention logits."""

    if not segments:
        return torch.zeros(query_length, context_length, device=device)

    tokens_per_frame = segments[0].tokens_per_frame
    if tokens_per_frame <= 0:
        raise ValueError("tokens_per_frame must be positive")
    if any(segment.tokens_per_frame != tokens_per_frame for segment in segments):
        raise ValueError("All Prompt Relay segments must use the same token grid")

    query_frames = (
        torch.arange(query_length, device=device, dtype=torch.long)
        + query_token_offset
    ) // tokens_per_frame
    penalty = torch.zeros(
        query_length, context_length, device=device, dtype=torch.float32
    )

    for segment in segments:
        if not segment.token_indices:
            raise ValueError("A Prompt Relay segment has no text tokens")
        if (
            min(segment.token_indices) < 0
            or max(segment.token_indices) >= context_length
        ):
            raise ValueError("A Prompt Relay token index is outside the text context")

        distance = (query_frames.float() - float(segment.midpoint)).abs()
        if segment.mode == "gaussian":
            cost = torch.relu(distance - segment.inner_window).square()
            cost = cost / (2.0 * segment.sigma**2)
        elif segment.mode == "hard":
            inside_interval = (
                (query_frames >= segment.frame_start)
                & (query_frames < segment.frame_end)
            )
            cost = torch.where(
                inside_interval,
                torch.zeros_like(distance),
                torch.full_like(distance, 1e9),
            )
        else:
            raise ValueError(f"Unsupported Prompt Relay mode: {segment.mode}")

        token_indices = torch.as_tensor(
            segment.token_indices, device=device, dtype=torch.long
        )
        penalty[:, token_indices] = cost[:, None]

    return penalty
