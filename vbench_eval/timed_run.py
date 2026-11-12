#!/usr/bin/env python
"""Run a generation command, time it, and record the time next to the video it makes.

    python vbench_eval/timed_run.py --video OUT.mp4 [--switch_latents 14] -- <generation command ...>

Writes OUT.meta.json beside OUT.mp4: {"gen_time_s": ..., "command": ..., "switch_frames": ...}.
evaluate.py reads gen_time_s from it (metric gen_time_s) and, when given, the exact frame(s) where the
prompt switches (the first frame of segment 2, 3, ...).
The time is wall-clock for the whole command, model loading included, so compare methods only when
they run with the same GPUs and the same speed flags (--offload_model, --t5_cpu, ...).

--switch_latents: the latent index where each later prompt (or Img2) starts, for Wan's 4-frame latents.
Latent L >= 1 begins at video frame 4L - 3, so --switch_latents 14 records switch frame 53. This holds for
Visual Relay runs too (the anchor latent is dropped, 77 frames out, prompt 2 from frame 53).
--switch_frames gives frame numbers directly instead.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vbeval import video_io as V  # noqa: E402


def latent_to_frame(latent):
    if latent < 1:
        raise ValueError("a switch latent must be 1 or more (latent 0 is the first frame)")
    return 4 * latent - 3


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: timed_run.py --video OUT.mp4 [--switch_latents L ...] -- <command ...>")
    split = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="the video file the command writes")
    p.add_argument("--switch_latents", type=int, nargs="+", default=None,
                   help="latent where each later prompt (or Img2) starts; recorded as frame 4L-3")
    p.add_argument("--switch_frames", type=int, nargs="+", default=None,
                   help="first frame of segment 2, 3, ... (instead of --switch_latents)")
    args = p.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        raise SystemExit("no command after --")
    if args.switch_latents and args.switch_frames:
        raise SystemExit("give --switch_latents or --switch_frames, not both")

    start = time.time()
    code = subprocess.run(command).returncode
    seconds = time.time() - start
    if code != 0:
        print(f"[timed_run] command failed with exit code {code} after {seconds:.0f} s; nothing recorded")
        return code
    if not os.path.isfile(args.video):
        print(f"[timed_run] command finished but {args.video} does not exist; nothing recorded")
        return 1
    record = {"gen_time_s": round(seconds, 1), "command": " ".join(shlex.quote(c) for c in command),
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.switch_latents:
        record["switch_frames"] = [latent_to_frame(l) for l in args.switch_latents]
    elif args.switch_frames:
        record["switch_frames"] = list(args.switch_frames)
    path = V.write_sidecar(args.video, record)
    print(f"[timed_run] {seconds:.0f} s; recorded in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
