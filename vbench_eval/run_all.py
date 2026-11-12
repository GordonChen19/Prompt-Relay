#!/usr/bin/env python
"""Score every method folder under one directory with the same prompt JSON, then compare them.

    python vbench_eval/run_all.py --prompts PROMPTS.json --root ROOT_DIR --out OUT_DIR [evaluate.py options]

ROOT_DIR holds one folder per method, each with the same file names, e.g.
    outputs/wan22_t2v/p001.mp4   outputs/wan22_relay/p001.mp4   outputs/minimax_h3/p001.mp4
Every folder is scored by evaluate.py (its own process, so GPU memory is freed between methods) into
OUT_DIR/<folder>/, then OUT_DIR/comparison.csv and comparison.md put the methods side by side.
Adding a method = adding a folder under ROOT_DIR. Any option not listed below goes to evaluate.py.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vbeval import report as R  # noqa: E402


def method_folders(root, only=None):
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and any(f.lower().endswith(".mp4") for f in os.listdir(os.path.join(root, d))))
    if only:
        wanted = [m.strip() for m in only.split(",") if m.strip()]
        missing = [m for m in wanted if m not in names]
        if missing:
            raise SystemExit(f"no video folder(s) {missing} under {root}; found {names}")
        names = wanted
    return names


def n_videos(summary):
    """Videos behind a method's means (methods can cover different entries, e.g. text-only vs image prompts)."""
    return max([v.get("n", 0) for v in summary.values() if isinstance(v, dict)] or [0])


def compare(out_dir, methods):
    """comparison.csv / .md: one row per method, one column per metric (mean over its videos)."""
    summaries = {}
    for m in methods:
        path = os.path.join(out_dir, m, "summary.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                summaries[m] = json.load(f)
    keys = R.ordered({k for s in summaries.values() for k in s})
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "videos"] + keys)
        for m, s in summaries.items():
            w.writerow([m, n_videos(s)] + [_fmt(s.get(k, {}).get("mean")) for k in keys])
    lines = ["| method | videos | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for m, s in summaries.items():
        lines.append(f"| {m} | {n_videos(s)} | " + " | ".join(_fmt(s.get(k, {}).get("mean")) or "-" for k in keys)
                     + " |")
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summaries, keys


def _fmt(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="the prompt JSON shared by every method")
    p.add_argument("--root", required=True, help="folder with one sub-folder of videos per method")
    p.add_argument("--out", required=True, help="results go to OUT/<method>/ plus OUT/comparison.csv")
    p.add_argument("--methods", default=None, help="comma-separated folder names to score (default: all)")
    p.add_argument("--compare_only", action="store_true", help="only rebuild the comparison from saved results")
    args, passthrough = p.parse_known_args(argv)

    methods = method_folders(args.root, args.methods)
    if not methods:
        raise SystemExit(f"no folders with .mp4 files under {args.root}")
    os.makedirs(args.out, exist_ok=True)
    print(f"[run_all] {len(methods)} method(s): {', '.join(methods)}")
    failed = []
    if not args.compare_only:
        for m in methods:
            cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), "--prompts", args.prompts,
                   "--videos_dir", os.path.join(args.root, m), "--out", os.path.join(args.out, m),
                   "--method", m] + passthrough + ([] if "--skip_missing" in passthrough else ["--skip_missing"])
            print(f"[run_all] === {m} ===\n  " + " ".join(cmd))
            if subprocess.run(cmd).returncode != 0:
                failed.append(m)
    summaries, keys = compare(args.out, methods)
    print(f"[run_all] comparison of {len(summaries)} method(s): "
          f"{os.path.join(args.out, 'comparison.csv')} and comparison.md")
    if failed:
        print(f"[run_all] finished with errors in: {', '.join(failed)} (see their per_video.csv 'errors' column)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
