"""Timeline helpers for JSON-driven multi-anchor Visual Relay."""

from __future__ import annotations

from dataclasses import dataclass


def shifted_rope_positions(
    internal_latent_frames: int,
    temporary_internal_indices: tuple[int, ...] | list[int],
    anchor_rope_offset: float = 0.5,
) -> tuple[float, ...]:
    """Return compressed temporal coordinates for an expanded J timeline."""
    anchors = tuple(temporary_internal_indices)
    if internal_latent_frames <= 0:
        raise ValueError("internal_latent_frames must be positive")
    if any(left >= right for left, right in zip(anchors, anchors[1:])):
        raise ValueError("temporary anchor indices must be strictly increasing")
    if anchors and (anchors[0] <= 0 or anchors[-1] >= internal_latent_frames):
        raise ValueError("temporary anchors must be interior latent frames")
    if not 0.0 <= float(anchor_rope_offset) <= 1.0:
        raise ValueError("anchor_rope_offset must be in [0, 1]")

    anchor_set = set(anchors)
    anchors_before = 0
    positions = []
    for frame in range(internal_latent_frames):
        if frame in anchor_set:
            positions.append(
                frame - anchors_before - 1 + float(anchor_rope_offset)
            )
            anchors_before += 1
        else:
            positions.append(float(frame - anchors_before))
    return tuple(positions)


def ordinary_anchor_attention_regions(
    internal_latent_frames: int,
    temporary_internal_indices: tuple[int, ...] | list[int],
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Partition ordinary queries by their nearest visible anchor pair.

    Each tuple is ``(query_start, query_end, visible_anchors)`` in expanded
    latent-frame coordinates. Anchor query frames themselves are excluded.
    """
    anchors = tuple(temporary_internal_indices)
    if internal_latent_frames <= 0:
        raise ValueError("internal_latent_frames must be positive")
    if not anchors:
        return ((0, internal_latent_frames, ()),)
    if any(left >= right for left, right in zip(anchors, anchors[1:])):
        raise ValueError("temporary anchor indices must be strictly increasing")
    if anchors[0] <= 0 or anchors[-1] >= internal_latent_frames:
        raise ValueError("temporary anchors must be interior latent frames")

    regions = [(0, anchors[0], (anchors[0],))]
    regions.extend(
        (left + 1, right, (left, right))
        for left, right in zip(anchors, anchors[1:])
    )
    regions.append(
        (anchors[-1] + 1, internal_latent_frames, (anchors[-1],))
    )
    return tuple(region for region in regions if region[0] < region[1])


def anchor_suffix_attention_frames(
    internal_latent_frames: int,
    temporary_internal_indices: tuple[int, ...] | list[int],
    anchor_index: int,
) -> tuple[int, ...]:
    """Return an anchor itself and future ordinary frames, excluding anchors."""
    anchors = tuple(temporary_internal_indices)
    if anchor_index not in anchors:
        raise ValueError("anchor_index must identify a temporary anchor")
    anchor_set = set(anchors)
    return (anchor_index,) + tuple(
        frame
        for frame in range(anchor_index + 1, internal_latent_frames)
        if frame not in anchor_set
    )


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

