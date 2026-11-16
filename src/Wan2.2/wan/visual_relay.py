"""Timeline helpers for JSON-driven multi-anchor Visual Relay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualRelayLayout:
    """Map final output coordinates to an expanded temporary-anchor timeline."""

    output_latent_frames: int
    image_indices: tuple[int, ...]
    internal_frames: tuple[tuple[str, int], ...]
    temporary_internal_indices: tuple[int, ...]

    @classmethod
    def build(
        cls,
        output_latent_frames: int,
        image_indices: tuple[int, ...] | list[int],
    ) -> "VisualRelayLayout":
        indices = tuple(image_indices)
        if output_latent_frames <= 0:
            raise ValueError("output_latent_frames must be positive")
        if not indices or indices[0] != 0:
            raise ValueError("Visual Relay requires a primary image at index 0")
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError("image indices must be strictly increasing")
        if indices[-1] >= output_latent_frames:
            raise ValueError(
                "image_index values must be smaller than the output latent length"
            )

        temporary = set(indices[1:])
        frames: list[tuple[str, int]] = []
        temporary_internal_indices: list[int] = []
        for output_index in range(output_latent_frames):
            if output_index in temporary:
                temporary_internal_indices.append(len(frames))
                frames.append(("anchor", output_index))
            frames.append(("output", output_index))
        return cls(
            output_latent_frames=output_latent_frames,
            image_indices=indices,
            internal_frames=tuple(frames),
            temporary_internal_indices=tuple(temporary_internal_indices),
        )

    @property
    def internal_latent_frames(self) -> int:
        return len(self.internal_frames)

    @property
    def temporary_count(self) -> int:
        return len(self.temporary_internal_indices)

    @property
    def retained_internal_indices(self) -> tuple[int, ...]:
        temporary = set(self.temporary_internal_indices)
        return tuple(
            index for index in range(self.internal_latent_frames)
            if index not in temporary
        )

    def auxiliary_internal_indices(self, anchor_number: int) -> tuple[int, ...]:
        """Return a stream containing ordinary frames and exactly one anchor."""

        if not 0 <= anchor_number < self.temporary_count:
            raise IndexError("anchor_number is out of range")
        own_anchor = self.temporary_internal_indices[anchor_number]
        temporary = set(self.temporary_internal_indices)
        return tuple(
            index for index in range(self.internal_latent_frames)
            if index == own_anchor or index not in temporary
        )

    def compact_anchor_index(self, anchor_number: int) -> int:
        """Position of an anchor after other temporary anchors are removed."""

        own_anchor = self.temporary_internal_indices[anchor_number]
        return self.auxiliary_internal_indices(anchor_number).index(own_anchor)

    def history_output_indices(
        self, anchor_number: int, count: int = 4
    ) -> tuple[int, ...]:
        """Immediately preceding ordinary output frames for an anchor."""

        if count < 0:
            raise ValueError("history count must be non-negative")
        output_index = self.image_indices[anchor_number + 1]
        return tuple(range(max(0, output_index - count), output_index))

