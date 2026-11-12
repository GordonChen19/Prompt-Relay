"""Where a video's prompt switches, and which frames each metric sees.

Frame ranges are half-open [start, end), 0-based, in the video as saved.

  switch frames  the first frame of segment 2, 3, ...  Taken from, in order:
               1. the video's sidecar "<name>.meta.json" ("switch_frames"), written at generation time (exact);
               2. the entry's "switch_fractions" (fractions of the video);
               3. an even split: segment k starts at round(k * frames / segments)  (e.g. a 9-second
                  video with 3 prompts gives 3 seconds each).
  full         segment k: from its cut to the next cut.
  core         segment k without margin_s seconds on each side of an internal cut. Prompt Relay hands
               over softly, so the frames next to a switch belong to neither prompt. Used for text alignment.
  switch       margin around each internal cut, switch_window_s seconds each side. Used for the
               continuity scores at the hand-over.
"""

MIN_SEGMENT_FRAMES = 4


class CutError(ValueError):
    """The cut positions do not fit this video."""


def resolve_cut_frames(n_frames, n_segments, fractions=None, frames=None):
    """First frame of each segment after the first (n_segments - 1 values)."""
    if n_segments <= 1:
        if frames:
            raise CutError("switch_frames given for a video with a single segment")
        return []
    if frames is not None:
        if (not isinstance(frames, list) or len(frames) != n_segments - 1
                or not all(isinstance(c, int) and not isinstance(c, bool) for c in frames)):
            raise CutError(f"switch_frames must be a list of {n_segments - 1} integers, got {frames!r}")
        cuts = list(frames)
    elif fractions is not None:
        cuts = [int(round(f * n_frames)) for f in fractions]
    else:
        cuts = [int(round(k * n_frames / n_segments)) for k in range(1, n_segments)]
    bounds = [0] + cuts + [n_frames]
    for a, b in zip(bounds, bounds[1:]):
        if b - a < MIN_SEGMENT_FRAMES:
            raise CutError(f"prompt switches at {cuts} leave a segment shorter than {MIN_SEGMENT_FRAMES} frames "
                           f"in a {n_frames}-frame video")
    return cuts


def plan_windows(n_frames, fps, cuts, margin_s=0.5, switch_window_s=0.5, min_core_frames=8):
    """Segments (full and core ranges) and switch windows for one video."""
    margin = int(round(margin_s * fps))
    half = max(1, int(round(switch_window_s * fps)))
    bounds = [0] + list(cuts) + [n_frames]
    n = len(bounds) - 1
    segments = []
    for k in range(n):
        start, end = bounds[k], bounds[k + 1]
        core = (start, end)
        for trim in range(margin, -1, -1):
            cs = start + (trim if k > 0 else 0)
            ce = end - (trim if k < n - 1 else 0)
            if ce - cs >= min_core_frames:
                core = (cs, ce)
                break
        segments.append({"index": k + 1, "full": (start, end), "core": core})
    switches = []
    for j, cut in enumerate(cuts):
        switches.append({"index": j + 1, "cut": cut,
                         "window": (max(0, cut - half), min(n_frames, cut + half))})
    return segments, switches


def describe(segments, switches, fps):
    """One line per range, for logs and --dry_run."""
    lines = []
    for s in segments:
        lines.append("  segment %d: frames %d-%d (core %d-%d)  %.2f-%.2f s" % (
            s["index"], s["full"][0], s["full"][1] - 1, s["core"][0], s["core"][1] - 1,
            s["full"][0] / fps, s["full"][1] / fps))
    for w in switches:
        lines.append("  switch %d at frame %d: window frames %d-%d" % (
            w["index"], w["cut"], w["window"][0], w["window"][1] - 1))
    return lines
