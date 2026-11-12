#!/usr/bin/env python
"""Score one method's videos: one prompt JSON + one folder of videos -> VBench (+ extra) scores.

    python vbench_eval/evaluate.py --prompts PROMPTS.json --videos_dir DIR --out OUT_DIR [options]

See vbench_eval/README.md for the JSON format, how videos are cut, and every option.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vbeval import metrics_extra as X  # noqa: E402
from vbeval import metrics_vbench as MV  # noqa: E402
from vbeval import report as R  # noqa: E402
from vbeval import segments as G  # noqa: E402
from vbeval import suite as S  # noqa: E402
from vbeval import video_io as V  # noqa: E402

EXTRA_METRICS = ["clip_t", "prompt_match", "clip_f", "gen_time"]
ALL_METRICS = MV.ALL_DIMS + EXTRA_METRICS
TEXT_METRICS = {"overall_consistency", "clip_t", "prompt_match"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="the prompt JSON (one entry per video; see README)")
    p.add_argument("--videos_dir", default=None,
                   help="folder holding the videos; overrides every entry's 'file_dir'")
    p.add_argument("--out", required=True, help="output folder for this method's scores")
    p.add_argument("--method", default=None, help="name stored in the results (default: the videos folder name)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vbench_root", default=MV.default_vbench_root(),
                   help="VBench git clone (default: $VBENCH_ROOT, else the submodule vbench_eval/third_party/VBench)")
    p.add_argument("--vbench_local", action="store_true",
                   help="VBench 'local' mode: download model files into ~/.cache/vbench instead of torch.hub")
    g = p.add_argument_group("cutting")
    g.add_argument("--margin_s", type=float, default=0.5,
                   help="seconds left out of each segment next to a switch for text alignment (default 0.5)")
    g.add_argument("--switch_window_s", type=float, default=0.5,
                   help="seconds either side of each switch for the hand-over scores (default 0.5)")
    g.add_argument("--min_core_frames", type=int, default=8,
                   help="the margin shrinks if a segment core would be shorter than this (default 8)")
    m = p.add_argument_group("metrics (VBench and VBench++ run by default; extras can be switched off)")
    m.add_argument("--only", default=None,
                   help="comma-separated list of metrics to run, e.g. 'temporal_flickering,clip_f' "
                        "(names: " + ", ".join(ALL_METRICS) + ")")
    m.add_argument("--skip", default=None, help="comma-separated list of metrics to leave out")
    m.add_argument("--no_clip_t", action="store_true", help="extra: no CLIP-T per segment (also no prompt_match)")
    m.add_argument("--no_prompt_match", action="store_true", help="extra: no prompt-matching accuracy")
    m.add_argument("--no_clip_f", action="store_true", help="extra: no CLIP-F adjacent-frame similarity")
    m.add_argument("--no_gen_time", action="store_true", help="extra: do not read generation time from sidecars")
    m.add_argument("--no_i2v", action="store_true", help="VBench++: no i2v_subject / i2v_background")
    m.add_argument("--clip_model", default="ViT-L/14", help="CLIP model for clip_t / clip_f (default ViT-L/14)")
    r = p.add_argument_group("running")
    r.add_argument("--dry_run", action="store_true",
                   help="check the JSON, find the videos, print the prompt switches and clips; load no models (no GPU)")
    r.add_argument("--force", action="store_true", help="rescore videos already in OUT_DIR/per_video.json")
    r.add_argument("--strict", action="store_true", help="stop at the first error instead of recording it")
    r.add_argument("--skip_missing", action="store_true",
                   help="leave out entries whose video is not in the folder (a method not run on them) instead "
                        "of counting them as errors; run_all.py sets this")
    r.add_argument("--delete_clips", action="store_true",
                   help="delete OUT_DIR/clips after scoring (clips are kept by default for inspection)")
    return p.parse_args(argv)


def enabled_metrics(args):
    names = set(ALL_METRICS)
    if args.only:
        names = {x.strip() for x in args.only.split(",") if x.strip()}
    for flag, metric in (("no_clip_t", "clip_t"), ("no_clip_t", "prompt_match"),
                         ("no_prompt_match", "prompt_match"), ("no_clip_f", "clip_f"),
                         ("no_gen_time", "gen_time"), ("no_i2v", "i2v_subject"), ("no_i2v", "i2v_background")):
        if getattr(args, flag):
            names.discard(metric)
    if args.skip:
        names -= {x.strip() for x in args.skip.split(",")}
    unknown = names - set(ALL_METRICS)
    if unknown:
        raise SystemExit(f"unknown metric(s): {', '.join(sorted(unknown))}. Known: {', '.join(ALL_METRICS)}")
    return {d for d in names if d in MV.ALL_DIMS}, {x for x in names if x in EXTRA_METRICS}


def fingerprint(entry, path, sidecar, settings):
    st = os.stat(path)
    blob = json.dumps([entry, os.path.abspath(path), st.st_size, int(st.st_mtime), sidecar, settings],
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def prepare_video(entry, args, dims, extras, out_dir):
    """Read one video, work out where its prompt switches and its windows are, write the clips VBench needs."""
    path = entry["video_path"]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"video not found: {path}")
    sidecar = V.read_sidecar(path)
    frames, fps = V.read_video(path)
    n_frames = len(frames)
    source = ("sidecar switch_frames" if sidecar.get("switch_frames") is not None else
              "entry switch_fractions" if entry["switch_fractions"] is not None else "even split")
    switch_frames = G.resolve_cut_frames(n_frames, entry["n_segments"], fractions=entry["switch_fractions"],
                                         frames=sidecar.get("switch_frames"))
    segments, switches = G.plan_windows(n_frames, fps, switch_frames, args.margin_s, args.switch_window_s,
                                        args.min_core_frames)
    clip_dir = os.path.join(out_dir, "clips", entry["id"])
    images = list(entry["images"])
    if dims & set(MV.I2V_DIMS):
        for k, image in enumerate(images, start=1):
            if image is None:
                continue
            if not os.path.isfile(image):
                raise FileNotFoundError(f"image {k} not found: {image}")
            if not args.dry_run:
                images[k - 1] = V.save_rgb_image(image, os.path.join(clip_dir, f"img{k}.png"))
    video = {
        "id": entry["id"], "path": os.path.abspath(path), "n_frames": n_frames, "fps": fps,
        "switch_frames": switch_frames, "switch_source": source, "segments": segments, "switches": switches,
        "whole_text": S.whole_video_text(entry), "seg_texts": S.segment_texts(entry),
        "images": images, "clips": {}, "sidecar": sidecar,
    }
    needed = MV.clips_needed(video, dims)
    for name, (a, b) in needed.items():
        clip_path = os.path.abspath(os.path.join(clip_dir, f"{name}.mp4"))
        if not args.dry_run:
            V.write_clip(frames[a:b], fps, clip_path)
        video["clips"][name] = clip_path
    warnings = []
    if (dims | extras) & TEXT_METRICS:
        texts = [("global prompt", video["whole_text"])] + [
            (f"local prompt {k}", t) for k, t in enumerate(video["seg_texts"], start=1)]
        warnings += X.truncation_warnings(texts)
    if "gen_time" in extras and "gen_time_s" not in sidecar:
        warnings.append("no gen_time_s in " + os.path.basename(V.sidecar_path(path)))
    video["warnings"] = warnings
    return video


def extra_scores(video, extras, clip_scorer):
    """CLIP-T / prompt matching / CLIP-F / generation time for one video."""
    out = {}
    if "gen_time" in extras:
        out["gen_time_s"] = float(video["sidecar"].get("gen_time_s", float("nan")))
    if not (extras & {"clip_t", "prompt_match", "clip_f"}) or clip_scorer is None:
        return out
    frames, _ = V.read_video(video["path"])
    if clip_scorer is not None:
        emb = clip_scorer.frames(frames)
        if "clip_t" in extras or "prompt_match" in extras:
            if "clip_t" in extras:
                whole = X.normalise(clip_scorer.texts([video["whole_text"]]))
                out["clip_t_global"] = float((X.normalise(emb) @ whole.T).mean())
            if video["seg_texts"]:
                matrix, own = X.clip_t_scores(emb, clip_scorer.texts(video["seg_texts"]), video["segments"])
                if "clip_t" in extras:
                    for k, v in enumerate(own, start=1):
                        out[f"clip_t_seg{k}"] = v
                    out["clip_t_seg_mean"] = R.mean(own)
                if "prompt_match" in extras and len(video["seg_texts"]) > 1:
                    out["prompt_match_acc"] = X.prompt_match_accuracy(matrix)
                    out["prompt_match_matrix"] = matrix.tolist()
        if "clip_f" in extras:
            out.update(X.clip_f_scores(emb, video["switches"]))
    return out


def main(argv=None):
    args = parse_args(argv)
    dims, extras = enabled_metrics(args)
    entries = S.load_suite(args.prompts, args.videos_dir)
    method = args.method or os.path.basename(os.path.normpath(
        args.videos_dir or os.path.dirname(entries[0]["video_path"]) or "."))
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    settings = {"margin_s": args.margin_s, "switch_window_s": args.switch_window_s,
                "min_core_frames": args.min_core_frames, "metrics": sorted(dims | extras),
                "clip_model": args.clip_model}
    previous = {} if (args.force or args.dry_run) else R.load_records(out_dir)
    print(f"[vbench_eval] method '{method}': {len(entries)} video(s) in {args.prompts}; "
          f"metrics: {', '.join(sorted(dims | extras)) or 'none'}")

    videos, records, plan, not_generated = [], {}, [], []
    for entry in entries:
        if args.skip_missing and not os.path.isfile(entry["video_path"]):
            not_generated.append(entry["id"])
            continue
        base = {"id": entry["id"], "method": method, "video": entry["video_path"],
                "global_prompt": entry["global_prompt"], "local_prompts": entry["local_prompts"],
                "images": entry["images"], "extra": entry["extra"]}
        try:
            fp = None
            if os.path.isfile(entry["video_path"]):
                fp = fingerprint(entry, entry["video_path"], V.read_sidecar(entry["video_path"]), settings)
                old = previous.get(entry["id"])
                if old and old.get("fingerprint") == fp and not old.get("errors"):
                    records[entry["id"]] = old
                    print(f"  {entry['id']}: already scored, kept (use --force to rescore)")
                    continue
            video = prepare_video(entry, args, dims, extras, out_dir)
            video["base"], video["fingerprint"] = base, fp
            lines = [f"  {entry['id']}: {video['n_frames']} frames at {video['fps']:g} fps "
                     f"({video['n_frames'] / video['fps']:.2f} s), {len(video['segments'])} segment(s), "
                     + (f"prompt switch at frame(s) {video['switch_frames']} ({video['switch_source']})"
                        if video["switch_frames"] else "no prompt switch")]
            lines += G.describe(video["segments"], video["switches"], video["fps"])
            lines += ["  clips: " + (", ".join(sorted(video["clips"])) or "none")]
            lines += [f"  note: {w}" for w in video["warnings"]]
            print("\n".join(lines))
            plan.append({k: video[k] for k in ("id", "path", "n_frames", "fps", "switch_frames", "switch_source",
                                               "segments", "switches", "clips", "images", "warnings")})
            videos.append(video)
        except (OSError, ValueError, G.CutError) as err:
            if args.strict:
                raise
            print(f"  {entry['id']}: SKIPPED - {err}")
            records[entry["id"]] = dict(base, errors={"video": str(err)}, metrics={})

    if args.dry_run:
        R.write_json(os.path.join(out_dir, "plan.json"), plan)
        print(f"[vbench_eval] dry run: {len(videos)} video(s) ready, {len(entries) - len(videos)} skipped. "
              f"Plan saved to {os.path.join(out_dir, 'plan.json')}")
        return 0 if len(videos) == len(entries) else 1

    timings, errors = {}, {v["id"]: {} for v in videos}
    vb = {v["id"]: {} for v in videos}
    work_dir = os.path.join(out_dir, "vbench_info")
    if dims and videos:
        MV.import_vbench(args.vbench_root)
    for dim in [d for d in MV.ALL_DIMS if d in dims]:
        jobs = [(v["id"], slot, clip, prompt, image)
                for v in videos for slot, clip, prompt, image in MV.build_jobs(v, dim)]
        if not jobs:
            continue
        print(f"[vbench_eval] {dim}: {len(jobs)} clip(s)")
        t0 = time.time()
        try:
            scores = MV.run_dimension(dim, jobs, work_dir, args.device, local=args.vbench_local)
        except Exception as err:  # noqa: BLE001 - one broken dimension should not lose the others
            if args.strict:
                raise
            traceback.print_exc()
            for v in videos:
                errors[v["id"]][dim] = f"{type(err).__name__}: {err}"
            continue
        timings[dim] = round(time.time() - t0, 1)
        for (vid, slot), score in scores.items():
            vb[vid].setdefault(dim, {})[slot] = score

    clip_scorer = None
    if videos and extras & {"clip_t", "prompt_match", "clip_f"}:
        try:
            clip_scorer = X.ClipScorer(args.clip_model, args.device)
        except Exception as err:  # noqa: BLE001
            if args.strict:
                raise
            traceback.print_exc()
            for v in videos:
                errors[v["id"]]["clip"] = f"{type(err).__name__}: {err}"
    t0 = time.time()
    for v in videos:
        try:
            ex = extra_scores(v, extras, clip_scorer)
        except Exception as err:  # noqa: BLE001
            if args.strict:
                raise
            traceback.print_exc()
            errors[v["id"]]["extras"] = f"{type(err).__name__}: {err}"
            ex = {}
        records[v["id"]] = dict(
            v["base"], fingerprint=v["fingerprint"], n_frames=v["n_frames"], fps=v["fps"],
            switch_frames=v["switch_frames"], switch_source=v["switch_source"],
            segments=[{"index": s["index"], "frames": list(s["full"]), "core": list(s["core"])}
                      for s in v["segments"]],
            switch_windows=[list(w["window"]) for w in v["switches"]],
            metrics=R.flatten(vb[v["id"]], ex),
            detail={"vbench_slots": vb[v["id"]],
                    "prompt_match_matrix": ex.get("prompt_match_matrix")},
            warnings=v["warnings"], errors=errors[v["id"]])
    if videos and extras - {"gen_time"}:
        timings["extras"] = round(time.time() - t0, 1)

    if not_generated:
        print(f"  no video for {len(not_generated)} entr{'y' if len(not_generated) == 1 else 'ies'} "
              f"(left out): {', '.join(not_generated)}")
    all_records = [records[e["id"]] for e in entries if e["id"] in records]
    summary = R.summarize([r for r in all_records if r.get("metrics")])
    vbench_root = args.vbench_root
    if not vbench_root and "vbench" in sys.modules:
        vbench_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["vbench"].__file__)))
    run_info = {"method": method, "prompts": os.path.abspath(args.prompts), "settings": settings,
                "videos": len(entries), "not_generated": not_generated, "scored": len(videos), "kept": len(all_records) - len(videos),
                "seconds_per_metric": timings, "vbench_commit": git_commit(vbench_root) if vbench_root else None,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    R.write_outputs(out_dir, all_records, summary, run_info)
    if args.delete_clips:
        shutil.rmtree(os.path.join(out_dir, "clips"), ignore_errors=True)
    print(R.format_summary(summary, f"[vbench_eval] {method}: mean over {len(all_records)} video(s)"))
    failed = {r["id"]: r["errors"] for r in all_records if r.get("errors")}
    for vid, errs in failed.items():
        print(f"  errors in {vid}: {errs}")
    print(f"[vbench_eval] results in {out_dir}: per_video.csv, summary.csv (and .json)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
