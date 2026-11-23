"""Wan-compatible Prompt Relay configuration for MiniMax H3."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_RELAY_MODES = frozenset({"gaussian", "hard"})


@dataclass(frozen=True)
class PromptRelayConfig:
    """Validated Prompt Relay input using the same schema as the Wan port."""

    global_prompt: str
    local_prompts: tuple[str, ...]
    segment_lengths: tuple[int, ...] = ()
    explicit_frame_intervals: tuple[tuple[int, int], ...] = ()
    mode: str = "gaussian"
    sigma: float = 0.932
    interval_expansion: int = 2

    @classmethod
    def from_json(cls, path: str | Path) -> "PromptRelayConfig":
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
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
        segment_lengths = data.get("segment_lengths", ())
        frame_intervals = data.get("frame_intervals", ())
        mode = data.get("mode", "gaussian")
        sigma = data.get("sigma", 0.932)
        interval_expansion = data.get("interval_expansion", 2)

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
        if (
            isinstance(interval_expansion, bool)
            or not isinstance(interval_expansion, int)
            or interval_expansion < 0
        ):
            raise ValueError("interval_expansion must be a non-negative integer")

        return cls(
            global_prompt=global_prompt.strip(),
            local_prompts=normalized_prompts,
            segment_lengths=normalized_lengths,
            explicit_frame_intervals=normalized_intervals,
            mode=mode.lower(),
            sigma=float(sigma),
            interval_expansion=interval_expansion,
        )

    @property
    def full_prompt_and_spans(self) -> tuple[str, tuple[tuple[int, int], ...]]:
        """Return the complete text and character spans for each local prompt."""

        text = self.global_prompt
        spans = []
        for prompt in self.local_prompts:
            text += " "
            start = len(text)
            text += prompt
            spans.append((start, len(text)))
        return text, tuple(spans)

    @property
    def full_prompt(self) -> str:
        return self.full_prompt_and_spans[0]

    def frame_intervals(self, internal_frames: int) -> tuple[tuple[int, int], ...]:
        """Resolve and symmetrically expand H3 transformer-frame intervals."""

        if internal_frames <= 0:
            raise ValueError("internal_frames must be positive")
        if self.explicit_frame_intervals:
            if any(end > internal_frames for _, end in self.explicit_frame_intervals):
                raise ValueError(
                    "frame_intervals cannot extend beyond the number of internal "
                    f"frames ({internal_frames})"
                )
            intervals = self.explicit_frame_intervals

        elif self.segment_lengths:
            total = sum(self.segment_lengths)
            if total != internal_frames:
                raise ValueError(
                    "segment_lengths must sum to the number of internal frames "
                    f"({internal_frames}), but sum to {total}"
                )
            boundaries = [0]
            for length in self.segment_lengths:
                boundaries.append(boundaries[-1] + length)
            intervals = tuple(zip(boundaries, boundaries[1:]))
        else:
            if len(self.local_prompts) > internal_frames:
                raise ValueError(
                    "There cannot be more local prompts than internal frames "
                    f"({len(self.local_prompts)} prompts, {internal_frames} frames)"
                )
            step = math.ceil(internal_frames / len(self.local_prompts))
            intervals = tuple(
                (step * index, min(step * (index + 1), internal_frames))
                for index in range(len(self.local_prompts))
            )

        expansion = self.interval_expansion
        return tuple(
            (max(0, start - expansion), min(internal_frames, end + expansion))
            for start, end in intervals
        )

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "global_prompt": self.global_prompt,
            "local_prompts": list(self.local_prompts),
            "mode": self.mode,
            "sigma": self.sigma,
            "interval_expansion": self.interval_expansion,
        }
        if self.explicit_frame_intervals:
            data["frame_intervals"] = [
                list(interval) for interval in self.explicit_frame_intervals
            ]
        elif self.segment_lengths:
            data["segment_lengths"] = list(self.segment_lengths)
        return data


@dataclass(frozen=True)
class ResolvedPromptRelaySchedule:
    intervals: tuple[tuple[int, int], ...]
    midpoints: tuple[int, ...]
    inner_windows: tuple[int, ...]
    mode: str
    sigma: float


def resolve_schedule(
    config: PromptRelayConfig, internal_frames: int
) -> ResolvedPromptRelaySchedule:
    intervals = config.frame_intervals(internal_frames)
    return ResolvedPromptRelaySchedule(
        intervals=intervals,
        midpoints=tuple((start + end) // 2 for start, end in intervals),
        inner_windows=tuple(
            (end - start) // 2 - 2 for start, end in intervals
        ),
        mode=config.mode,
        sigma=config.sigma,
    )
