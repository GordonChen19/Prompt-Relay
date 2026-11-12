#!/usr/bin/env python
"""Offline checks for vbench_eval: no GPU, no VBench models, no torch needed.

    python vbench_eval/tests/test_offline.py [--vbench_root PATH_TO_VBENCH_CLONE (default: the submodule third_party/VBench)]

Checks the JSON rules, the cutting, which clips each metric scores, the score bookkeeping, the output
files, caching, the metric flags, run_all.py and timed_run.py, with stand-in models on small synthetic
videos. With --vbench_root it also runs VBench's own JSON readers on the files we hand to VBench; with the
Wan, Visual Relay and MiniMax H3 code present it checks every generation command against their own parsers.
Needs numpy, opencv-python, imageio, imageio-ffmpeg and Pillow.
"""
import argparse
import ast
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import evaluate as E  # noqa: E402
import generate_suite as GS  # noqa: E402
import run_all as RA  # noqa: E402
import timed_run as TR  # noqa: E402
from vbeval import metrics_extra as X  # noqa: E402
from vbeval import metrics_vbench as MV  # noqa: E402
from vbeval import report as R  # noqa: E402
from vbeval import segments as G  # noqa: E402
from vbeval import suite as S  # noqa: E402
from vbeval import video_io as V  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail and not ok else ""))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def make_video(path, n=81, fps=16, cut_at=None, size=(64, 48)):
    """Smooth gradient drift; a different colour scheme from cut_at (a hard cut)."""
    w, h = size
    frames = []
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(n):
        base = (xx * 3 + yy * 2 + t * 2) % 255
        frame = np.stack([base, (base + 60) % 255, (255 - base)], -1).astype(np.uint8)
        if cut_at is not None and t >= cut_at:
            frame = (255 - frame).copy()
        frames.append(frame)
    V.write_clip(np.stack(frames), fps, path)
    return path


def make_image(path, color):
    from PIL import Image
    Image.new("RGBA", (40, 30), color + (128,)).save(path)
    return path


# ------------------------------------------------------------------ fakes standing in for the models
class FakeClip:
    """Frame embedding = mean colour; text embedding fixed per prompt."""
    calls = 0

    def __init__(self, name, device):
        FakeClip.calls += 1

    def frames(self, frames):
        return np.stack([f.reshape(-1, 3).mean(0) + 1.0 for f in frames])

    def texts(self, texts):
        return np.stack([np.array([1.0 + i, 2.0, 3.0 - i]) for i in range(len(texts))])


def fake_run_dimension(dim, jobs, work_dir, device, local=False):
    """Score = clip length / 1000 + slot code, so every slot can be traced back."""
    MV.write_full_info(dim, jobs, os.path.join(work_dir, f"{dim}_full_info.json"))
    out = {}
    for vid, slot, clip, prompt, image in jobs:
        frames, _ = V.read_video(clip)
        out[(vid, slot)] = len(frames) / 1000.0 + (0.5 if image else 0.0)
    return out


@contextlib.contextmanager
def fakes():
    saved = (MV.run_dimension, MV.import_vbench, X.ClipScorer)
    MV.run_dimension, MV.import_vbench = fake_run_dimension, (lambda root=None: None)
    X.ClipScorer = FakeClip
    try:
        yield
    finally:
        MV.run_dimension, MV.import_vbench, X.ClipScorer = saved


def run_quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = fn(*a, **k)
    return code, buf.getvalue()


# ------------------------------------------------------------------ tests
def test_suite(tmp):
    good = write_json(os.path.join(tmp, "good.json"), [
        {"file_name": "a.mp4", "file_dir": "x", "global_prompt": "g", "local_prompts": ["p1", "p2"],
         "images": ["i1.png"], "switch_fractions": [0.6], "tags": ["t"]},
        {"id": "b", "file_name": "b.mp4", "global_prompt": "only global", "file_dir": "x"}])
    entries = S.load_suite(good)
    a, b = entries
    check("suite: a list of entries loads; id defaults to the file name; images padded to one per segment; "
          "unknown fields kept as extra",
          a["id"] == "a" and a["n_segments"] == 2 and a["images"] == ["i1.png", None]
          and a["switch_fractions"] == [0.6] and a["extra"] == {"tags": ["t"]} and b["n_segments"] == 1
          and a["video_path"] == os.path.join("x", "a.mp4"))
    one = write_json(os.path.join(tmp, "one.json"), {"file_name": "c.mp4", "global_prompt": "g"})
    check("suite: a single object loads; --videos_dir overrides file_dir",
          S.load_suite(one, videos_dir="vids")[0]["video_path"] == os.path.join("vids", "c.mp4")
          and S.load_suite(good, videos_dir="vids")[0]["video_path"] == os.path.join("vids", "a.mp4"))
    wrapped = write_json(os.path.join(tmp, "wrapped.json"), {"prompts": [{"file_name": "c.mp4", "file_dir": "d",
                                                                          "global_prompt": "g"}]})
    check("suite: {'prompts': [...]} loads", len(S.load_suite(wrapped)) == 1)
    bad_cases = {
        "no file_name": {"global_prompt": "g", "file_dir": "x"},
        "not mp4": {"file_name": "a.avi", "global_prompt": "g", "file_dir": "x"},
        "path in file_name": {"file_name": "x/a.mp4", "global_prompt": "g"},
        "no folder": {"file_name": "a.mp4", "global_prompt": "g"},
        "no text": {"file_name": "a.mp4", "file_dir": "x"},
        "empty local prompt": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p", " "]},
        "too many images": {"file_name": "a.mp4", "file_dir": "x", "global_prompt": "g",
                            "local_prompts": ["p1", "p2"], "images": ["1", "2", "3"]},
        "switch_fractions wrong count": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p1", "p2", "p3"],
                                         "switch_fractions": [0.5]},
        "switch_fractions not increasing": {"file_name": "a.mp4", "file_dir": "x",
                                            "local_prompts": ["p1", "p2", "p3"], "switch_fractions": [0.6, 0.4]},
        "switch_fractions outside 0-1": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p1", "p2"],
                                         "switch_fractions": [1.0]},
        "switch_fractions on one segment": {"file_name": "a.mp4", "file_dir": "x", "global_prompt": "g",
                                            "switch_fractions": [0.5]},
        "old name cuts": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p1", "p2"], "cuts": [0.5]},
        "image_latents count": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p1", "p2"],
                                "images": ["i1.png", "i2.png"], "image_latents": [0]},
        "image_latents not from 0": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p1", "p2"],
                                     "images": ["i1.png", "i2.png"], "image_latents": [1, 14]},
        "image_latents not increasing": {"file_name": "a.mp4", "file_dir": "x", "local_prompts": ["p1", "p2"],
                                         "images": ["i1.png", "i2.png"], "image_latents": [0, 0]},
    }
    rejected = [name for name, entry in bad_cases.items()
                if raises(lambda e=entry: S.load_suite(write_json(os.path.join(tmp, "bad.json"), [e])),
                          S.SuiteError)]
    dup = write_json(os.path.join(tmp, "dup.json"), [{"file_name": "a.mp4", "file_dir": "x", "global_prompt": "g"},
                                                     {"file_name": "a.mp4", "file_dir": "y", "global_prompt": "g"}])
    check(f"suite: {len(bad_cases) + 1} malformed files are rejected with a message",
          len(rejected) == len(bad_cases) and raises(lambda: S.load_suite(dup), S.SuiteError),
          f"accepted: {sorted(set(bad_cases) - set(rejected))}")
    check("suite: the whole video is compared with the global prompt, or the local prompts joined",
          S.whole_video_text(a) == "g" and S.whole_video_text(
              {"global_prompt": "", "local_prompts": ["p1", "p2"]}) == "p1 p2")


def test_segments():
    check("switches: even split (2 prompts in 81 frames -> 40; 3 in 77 -> 26, 51), sidecar frames win, then "
          "fractions; one segment has no switch",
          G.resolve_cut_frames(81, 2) == [40] and G.resolve_cut_frames(77, 3) == [26, 51]
          and G.resolve_cut_frames(77, 2, fractions=[0.68], frames=[53]) == [53]
          and G.resolve_cut_frames(77, 2, fractions=[0.68]) == [52] and G.resolve_cut_frames(81, 1) == [])
    check("switches: wrong-length, non-integer, out-of-range and too-short segments are refused",
          raises(lambda: G.resolve_cut_frames(77, 2, frames=[10, 20]), G.CutError)
          and raises(lambda: G.resolve_cut_frames(77, 2, frames=[52.5]), G.CutError)
          and raises(lambda: G.resolve_cut_frames(77, 2, frames=[76]), G.CutError)
          and raises(lambda: G.resolve_cut_frames(77, 2, frames=[2]), G.CutError)
          and raises(lambda: G.resolve_cut_frames(10, 4), G.CutError))
    segs, sw = G.plan_windows(77, 16, [53], margin_s=0.5, switch_window_s=0.5)
    check("windows: a Visual Relay video (77 frames, 16 fps, switch at 53): segment 1 = 0-52 with core 0-44, segment 2 = "
          "53-76 with core 61-76, switch window 45-60",
          [s["full"] for s in segs] == [(0, 53), (53, 77)] and [s["core"] for s in segs] == [(0, 45), (61, 77)]
          and sw == [{"index": 1, "cut": 53, "window": (45, 61)}], f"{segs} {sw}")
    segs3, _ = G.plan_windows(81, 16, [27, 54])
    check("windows: a middle segment loses the margin on both sides, the first and last on one side only",
          [s["core"] for s in segs3] == [(0, 19), (35, 46), (62, 81)])
    short, _ = G.plan_windows(30, 16, [12, 20], margin_s=0.5, min_core_frames=8)
    check("windows: the margin shrinks when a core would fall under min_core_frames (8-frame middle segment "
          "keeps all 8)", short[1]["core"] == (12, 20) and short[0]["core"][1] - short[0]["core"][0] >= 8,
          str(short))
    _, edge = G.plan_windows(40, 16, [5], switch_window_s=0.5)
    check("windows: switch windows stay inside the video", edge[0]["window"] == (0, 13))


def test_jobs():
    def video(n, texts, images, switches=True):
        segs, sw = G.plan_windows(81, 16, G.resolve_cut_frames(81, n))
        v = {"id": "v", "path": "/v.mp4", "segments": segs, "switches": sw if switches else [],
             "whole_text": "G", "seg_texts": texts, "images": images, "clips": {}}
        v["clips"] = {k: f"/c/{k}.mp4" for k in MV.clips_needed(v, set(MV.ALL_DIMS))}
        return v
    v2 = video(2, ["A", "B"], ["/i1.png", "/i2.png"])
    check("clips: a 2-prompt Visual Relay video needs both cores, both full segments and the switch window",
          sorted(v2["clips"]) == ["core_seg1", "core_seg2", "full_seg1", "full_seg2", "switch1"])
    jobs = {d: MV.build_jobs(v2, d) for d in MV.ALL_DIMS}
    check("jobs: overall_consistency = whole video with the global prompt + each core with its own prompt",
          jobs["overall_consistency"] == [("global", "/v.mp4", "G", None), ("seg1", "/c/core_seg1.mp4", "A", None),
                                          ("seg2", "/c/core_seg2.mp4", "B", None)])
    check("jobs: subject/background consistency = whole + each full segment + the switch window; flicker and "
          "motion smoothness = whole + switch; dynamic degree = whole + segments; aesthetic/imaging = whole",
          [j[0] for j in jobs["subject_consistency"]] == ["whole", "seg1", "seg2", "switch1"]
          and [j[0] for j in jobs["background_consistency"]] == ["whole", "seg1", "seg2", "switch1"]
          and [j[0] for j in jobs["temporal_flickering"]] == ["whole", "switch1"]
          and [j[0] for j in jobs["motion_smoothness"]] == ["whole", "switch1"]
          and [j[0] for j in jobs["dynamic_degree"]] == ["whole", "seg1", "seg2"]
          and [j[0] for j in jobs["aesthetic_quality"]] == ["whole"]
          and [j[0] for j in jobs["imaging_quality"]] == ["whole"])
    check("jobs: VBench++ compares image k with full segment k",
          jobs["i2v_subject"] == [("img1", "/c/full_seg1.mp4", "G", "/i1.png"),
                                  ("img2", "/c/full_seg2.mp4", "G", "/i2.png")]
          and [j[0] for j in jobs["i2v_background"]] == ["img1", "img2"])
    v1 = video(1, [], ["/i1.png"])
    j1 = {d: MV.build_jobs(v1, d) for d in MV.ALL_DIMS}
    check("jobs: a single-prompt I2V video is scored whole, with no clips, and its image against the whole video",
          v1["clips"] == {} and all(j[1] == "/v.mp4" for d in MV.ALL_DIMS for j in j1[d])
          and [j[0] for j in j1["overall_consistency"]] == ["global"] and len(j1["i2v_subject"]) == 1)
    v1t = video(1, ["only"], [None])
    check("jobs: one local prompt gives it a core clip so its score is kept apart from the global one",
          sorted(v1t["clips"]) == ["core_seg1"] and [j[0] for j in MV.build_jobs(v1t, "overall_consistency")]
          == ["global", "seg1"] and MV.build_jobs(v1t, "i2v_subject") == [])
    flick_only = MV.clips_needed(v2, {"temporal_flickering"})
    text_only = MV.clips_needed(v2, {"overall_consistency"})
    check("clips: only the clips the chosen metrics need are written",
          sorted(flick_only) == ["switch1"] and sorted(text_only) == ["core_seg1", "core_seg2"])


def test_vbench_contract(tmp, vbench_root):
    """Run VBench's own JSON readers on what write_full_info produces."""
    def load_fn(path, names):
        tree = ast.parse(open(path, encoding="utf-8").read())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {"os": os, "json": json}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
        return ns
    vb = load_fn(os.path.join(vbench_root, "vbench", "utils.py"), {"load_json", "load_dimension_info"})
    i2v = load_fn(os.path.join(vbench_root, "vbench2_beta_i2v", "utils.py"),
                  {"load_json", "load_i2v_dimension_info"})
    jobs = [("v", "global", "/v.mp4", "G", None), ("v", "seg1", "/c1.mp4", "A", None),
            ("w", "seg1", "/c2.mp4", "B", None)]
    path = MV.write_full_info("overall_consistency", jobs, os.path.join(tmp, "info", "oc.json"))
    videos, prompts = vb["load_dimension_info"](path, dimension="overall_consistency", lang="en")
    check("VBench contract: VBench's load_dimension_info reads our file as one clip per entry, each with its "
          "own prompt", videos == ["/v.mp4", "/c1.mp4", "/c2.mp4"]
          and [(p["prompt"], p["video_list"]) for p in prompts] == [("G", ["/v.mp4"]), ("A", ["/c1.mp4"]),
                                                                     ("B", ["/c2.mp4"])])
    ij = [("v", "img1", "/s1.mp4", "G", "/i1.png"), ("v", "img2", "/s2.mp4", "G", "/i2.png")]
    path = MV.write_full_info("i2v_subject", ij, os.path.join(tmp, "info", "i2v.json"))
    pairs, _ = i2v["load_i2v_dimension_info"](path, dimension="i2v_subject", lang="en", resolution="1-1")
    check("VBench contract: VBench-I2V's load_i2v_dimension_info pairs each clip with our image",
          pairs == [("/i1.png", "/s1.mp4"), ("/i2.png", "/s2.mp4")])
    src = open(os.path.join(vbench_root, "vbench", "__init__.py"), encoding="utf-8").read()
    utils_src = open(os.path.join(vbench_root, "vbench", "utils.py"), encoding="utf-8").read()
    i2v_src = open(os.path.join(vbench_root, "vbench2_beta_i2v", "utils.py"), encoding="utf-8").read()
    dims_ok = all(f"def compute_{d}(json_dir, device, submodules_list" in
                  open(os.path.join(vbench_root, "vbench" if d in MV.VBENCH_DIMS else "vbench2_beta_i2v",
                                    f"{d}.py"), encoding="utf-8").read() for d in MV.ALL_DIMS)
    check("VBench contract: every dimension we use has compute_<dim>(json_dir, device, submodules_list) and "
          "VBench.evaluate calls them the same way; init_submodules(dimension_list, local, read_frame) exists",
          dims_ok and "evaluate_func(cur_full_info_path, self.device, submodules_list" in src
          and "def init_submodules(dimension_list, local=False, read_frame=False)" in utils_src
          and "def init_submodules(dimension_list, local=False, read_frame=False" in i2v_src)
    iq = open(os.path.join(vbench_root, "vbench", "imaging_quality.py"), encoding="utf-8").read()
    check("VBench contract: imaging_quality reports per-video scores on 0-100 (we divide by 100)",
          "'video_results': acc_score_video/len(images)" in iq and "average_score / 100." in iq
          and MV.SCALE == {"imaging_quality": 0.01})


def test_extra_math():
    emb = np.zeros((10, 2))
    emb[:5] = [1, 0]
    emb[5:] = [0, 1]
    segs, _ = G.plan_windows(10, 4, [5], margin_s=0.0, min_core_frames=1)
    matrix, own = X.clip_t_scores(emb, np.array([[1.0, 0.0], [0.0, 1.0]]), segs)
    swapped, _ = X.clip_t_scores(emb, np.array([[0.0, 1.0], [1.0, 0.0]]), segs)
    check("CLIP-T: each segment is scored on its core frames with its own prompt; prompt matching is 1 in order "
          "and 0 when the prompts are swapped",
          np.allclose(matrix, [[1, 0], [0, 1]]) and own == [1.0, 1.0] and X.prompt_match_accuracy(matrix) == 1.0
          and X.prompt_match_accuracy(swapped) == 0.0 and math.isnan(X.prompt_match_accuracy(matrix[:1, :1])))
    check("prompt matching is judged per segment (row): segment 2 matches its own prompt even when prompt 1 "
          "fits segment 2 better than segment 1", X.prompt_match_accuracy([[0.5, 0.4], [0.6, 0.9]]) == 1.0)
    emb20 = np.zeros((20, 2))
    emb20[:8] = [1, 0]
    emb20[8:10] = [0, 1]
    emb20[10:] = [0, 1]
    segs20, _ = G.plan_windows(20, 4, [10], margin_s=0.5, min_core_frames=4)
    _, own20 = X.clip_t_scores(emb20, np.array([[1.0, 0.0], [0.0, 1.0]]), segs20)
    check("CLIP-T uses only the core frames: frames 8-9 (already showing event 2) do not count against "
          "segment 1", segs20[0]["core"] == (0, 8) and own20[0] == 1.0, f"{segs20} {own20}")
    _, sw = G.plan_windows(10, 4, [5], switch_window_s=0.5)
    cf = X.clip_f_scores(emb, sw)
    calm = X.clip_f_scores(np.ones((10, 2)), sw)
    check("CLIP-F: mean and minimum of adjacent-frame similarity; the jump at frame 5 shows as switch minimum 0",
          abs(cf["clip_f_mean"] - 8 / 9) < 1e-9 and cf["clip_f_min"] == 0.0 and cf["clip_f_switch_min"] == 0.0
          and abs(calm["clip_f_switch_min"] - 1.0) < 1e-9
          and "clip_f_switch_min" not in X.clip_f_scores(emb, []), str(cf))


def test_report(tmp):
    vb = {"overall_consistency": {"global": 0.2, "seg1": 0.3, "seg2": 0.1},
          "subject_consistency": {"whole": 0.9, "seg1": 0.95, "seg2": 0.85, "switch1": 0.7},
          "temporal_flickering": {"whole": 0.97, "switch1": 0.93},
          "dynamic_degree": {"whole": 1.0, "seg1": 1.0, "seg2": 0.0},
          "imaging_quality": {"whole": 0.6},
          "i2v_subject": {"img1": 0.8, "img2": 0.6}}
    m = R.flatten(vb, {"clip_t_seg_mean": 0.25, "gen_time_s": 600.0, "prompt_match_matrix": [[1]]})
    check("report: per-segment and switch scores become seg_mean / switch_min columns; lists stay out",
          m["overall_consistency_seg_mean"] == 0.2 and m["overall_consistency_global"] == 0.2
          and m["subject_consistency"] == 0.9 and abs(m["subject_consistency_seg_mean"] - 0.9) < 1e-12
          and m["subject_consistency_switch_min"] == 0.7 and m["temporal_flickering_switch_min"] == 0.93
          and m["dynamic_degree_seg_mean"] == 0.5 and m["imaging_quality"] == 0.6
          and m["i2v_subject_img2"] == 0.6 and m["gen_time_s"] == 600.0 and "prompt_match_matrix" not in m, str(m))
    two = R.flatten({"subject_consistency": {"whole": 0.9, "switch1": 0.7, "switch2": 0.9},
                     "motion_smoothness": {"whole": 0.99, "switch1": 0.98, "switch2": 0.95}}, {})
    check("report: with several switches the switch score is the lowest one",
          two["subject_consistency_switch_min"] == 0.7 and two["motion_smoothness_switch_min"] == 0.95)
    s = R.summarize([{"metrics": {"a": 1.0, "b": float("nan")}}, {"metrics": {"a": 3.0, "b": 2.0}}])
    check("report: summary = mean, std and count over videos, skipping missing values",
          s["a"] == {"mean": 2.0, "std": 1.0, "n": 2} and s["b"]["n"] == 1 and s["b"]["mean"] == 2.0)


def test_end_to_end(tmp):
    vids = os.path.join(tmp, "outputs", "relay")
    os.makedirs(vids)
    make_video(os.path.join(vids, "p1.mp4"), n=77, cut_at=53)
    make_video(os.path.join(vids, "p2.mp4"), n=81)
    make_video(os.path.join(vids, "p3.mp4"), n=81)
    V.write_sidecar(os.path.join(vids, "p1.mp4"), {"gen_time_s": 812.5, "switch_frames": [53]})
    img1, img2 = make_image(os.path.join(tmp, "i1.png"), (200, 10, 10)), make_image(os.path.join(tmp, "i2.png"),
                                                                                       (10, 10, 200))
    prompts = write_json(os.path.join(tmp, "prompts.json"), [
        {"id": "p1", "file_name": "p1.mp4", "global_prompt": "G1", "local_prompts": ["A", "B"],
         "images": [img1, img2]},
        {"id": "p2", "file_name": "p2.mp4", "global_prompt": "G2", "local_prompts": ["A", "B", "C"]},
        {"id": "p3", "file_name": "p3.mp4", "global_prompt": "just one prompt"},
        {"id": "missing", "file_name": "nope.mp4", "global_prompt": "G"}])
    out = os.path.join(tmp, "eval", "relay")
    with fakes():
        code, log = run_quiet(E.main, ["--prompts", prompts, "--videos_dir", vids, "--out", out])
    recs = {r["id"]: r for r in json.load(open(os.path.join(out, "per_video.json")))}
    p1, p2, p3 = recs["p1"]["metrics"], recs["p2"]["metrics"], recs["p3"]["metrics"]
    check("end to end: exit code 1 because one video is missing; it is recorded, the other three are scored",
          code == 1 and recs["missing"]["errors"] and not recs["missing"]["metrics"]
          and all(not recs[i]["errors"] for i in ("p1", "p2", "p3")), log[-2000:])
    check("end to end: the sidecar's switch frame 53 and generation time are used; p2 is split evenly at 27/54",
          recs["p1"]["switch_frames"] == [53] and recs["p1"]["switch_source"] == "sidecar switch_frames"
          and p1["gen_time_s"] == 812.5 and recs["p2"]["switch_frames"] == [27, 54]
          and recs["p2"]["switch_source"] == "even split" and "gen_time_s" in p2 and p2["gen_time_s"] is None)
    # fake VBench score = clip frames / 1000 (+0.5 with an image): traces which clip each slot used
    check("end to end: VBench slots got the right clips (p1: core 0-44 = 45 frames, core 61-76 = 16, segments "
          "53 + 24, switch 16; whole 77)",
          abs(p1["overall_consistency_seg1"] - 0.045) < 1e-9 and abs(p1["overall_consistency_seg2"] - 0.016) < 1e-9
          and abs(p1["overall_consistency_global"] - 0.077) < 1e-9
          and abs(p1["subject_consistency_seg1"] - 0.053) < 1e-9 and abs(p1["subject_consistency_seg2"] - 0.024)
          < 1e-9 and abs(p1["subject_consistency_switch_min"] - 0.016) < 1e-9
          and abs(p1["i2v_subject_img1"] - 0.553) < 1e-9 and abs(p1["i2v_background_img2"] - 0.524) < 1e-9, str(p1))
    check("end to end: a single-prompt video gets whole-video scores only (no segment, switch or match columns)",
          not any(k.endswith(("_seg1", "_seg_mean", "_switch_min")) or k == "prompt_match_acc" for k in p3)
          and "clip_t_global" in p3 and "overall_consistency_global" in p3)
    check("end to end: CLIP-T per segment, prompt matching and CLIP-F columns for relay videos; no shot-cut "
          "columns (TransNetV2 was removed)",
          "clip_t_seg1" in p1 and "prompt_match_acc" in p2 and "clip_f_switch_min" in p1
          and not any("cut" in k or "transnet" in k for k in p1), str(sorted(p1)))
    files_ok = all(os.path.isfile(os.path.join(out, f)) for f in
                   ("per_video.csv", "summary.csv", "summary.json", "run_info.json", "per_video.json"))
    header = open(os.path.join(out, "per_video.csv")).readline().strip().split(",")
    info = json.load(open(os.path.join(out, "run_info.json")))
    check("end to end: CSV/JSON outputs written, columns start with the text-alignment scores, timings recorded",
          files_ok and header[:6] == ["id", "video", "frames", "fps", "switch_frames", "errors"]
          and header[6] == "overall_consistency_seg_mean" and "overall_consistency" in info["seconds_per_metric"])
    full = json.load(open(os.path.join(out, "vbench_info", "overall_consistency_full_info.json")))
    check("end to end: VBench receives each clip with its own prompt",
          [(e["prompt_en"], os.path.basename(e["video_list"][0])) for e in full[:3]] ==
          [("G1", "p1.mp4"), ("A", "core_seg1.mp4"), ("B", "core_seg2.mp4")])
    with fakes():
        calls = FakeClip.calls
        code2, log2 = run_quiet(E.main, ["--prompts", prompts, "--videos_dir", vids, "--out", out])
    check("caching: a second run keeps already-scored videos and loads no models for them",
          log2.count("already scored, kept") == 3 and FakeClip.calls == calls, log2[-800:])
    V.write_sidecar(os.path.join(vids, "p3.mp4"), {"gen_time_s": 100.0})
    with fakes():
        _, log3 = run_quiet(E.main, ["--prompts", prompts, "--videos_dir", vids, "--out", out])
    recs3 = {r["id"]: r for r in json.load(open(os.path.join(out, "per_video.json")))}
    check("caching: a changed sidecar rescores just that video",
          log3.count("already scored, kept") == 2 and recs3["p3"]["metrics"]["gen_time_s"] == 100.0)
    out2 = os.path.join(tmp, "eval", "flags")
    with fakes():
        run_quiet(E.main, ["--prompts", prompts, "--videos_dir", vids, "--out", out2, "--no_clip_t", "--no_clip_f",
                           "--no_gen_time", "--no_i2v", "--delete_clips"])
    m = {r["id"]: r for r in json.load(open(os.path.join(out2, "per_video.json")))}["p1"]["metrics"]
    check("flags: --no_clip_t --no_clip_f --no_gen_time --no_i2v remove exactly those columns",
          not any(k.startswith(("clip_t", "prompt_match", "clip_f", "gen_time", "i2v"))
                  for k in m) and "overall_consistency_seg1" in m and "motion_smoothness" in m)
    check("flags: --delete_clips removes the clips after scoring",
          not os.path.exists(os.path.join(out2, "clips")) and os.path.isfile(os.path.join(out2, "summary.csv")))
    out3 = os.path.join(tmp, "eval", "only")
    with fakes():
        run_quiet(E.main, ["--prompts", prompts, "--videos_dir", vids, "--out", out3,
                           "--only", "temporal_flickering,clip_f", "--no_clip_f"])
    m3 = {r["id"]: r for r in json.load(open(os.path.join(out3, "per_video.json")))}["p1"]["metrics"]
    clips3 = sorted(os.listdir(os.path.join(out3, "clips", "p1")))
    check("flags: --only runs just those metrics (and --no_... still applies); only the clips they need are made",
          set(m3) == {"temporal_flickering", "temporal_flickering_switch_min"} and clips3 == ["switch1.mp4"],
          f"{sorted(m3)} {clips3}")
    check("flags: an unknown metric name stops with a message",
          raises(lambda: E.enabled_metrics(E.parse_args(["--prompts", "x", "--out", "y", "--only", "clipt"])),
                 SystemExit))
    out4 = os.path.join(tmp, "eval", "dry")
    code4, log4 = run_quiet(E.main, ["--prompts", prompts, "--videos_dir", vids, "--out", out4, "--dry_run"])
    plan = json.load(open(os.path.join(out4, "plan.json")))
    check("dry run: no models, no clips written; prints the switches and saves plan.json; exit 1 for the missing "
          "video", code4 == 1 and len(plan) == 3 and not os.path.isdir(os.path.join(out4, "clips"))
          and "prompt switch at frame(s) [53] (sidecar switch_frames)" in log4 and "SKIPPED" in log4)


def test_run_all_and_timer(tmp):
    root = os.path.join(tmp, "all_methods")
    for m in ("method_a", "method_b"):
        os.makedirs(os.path.join(root, m))
        make_video(os.path.join(root, m, "p1.mp4"), n=33)
    V.write_sidecar(os.path.join(root, "method_a", "p1.mp4"), {"gen_time_s": 10.0})
    V.write_sidecar(os.path.join(root, "method_b", "p1.mp4"), {"gen_time_s": 30.0})
    make_video(os.path.join(root, "method_b", "p2.mp4"), n=33)
    V.write_sidecar(os.path.join(root, "method_b", "p2.mp4"), {"gen_time_s": 50.0})
    os.makedirs(os.path.join(root, "empty_folder"))
    prompts = write_json(os.path.join(tmp, "p.json"), [{"file_name": "p1.mp4", "global_prompt": "G",
                                                       "local_prompts": ["A", "B"]},
                                                      {"file_name": "p2.mp4", "global_prompt": "G2"}])
    out = os.path.join(tmp, "eval_all")
    code, log = run_quiet(RA.main, ["--prompts", prompts, "--root", root, "--out", out, "--only", "gen_time"])
    table = open(os.path.join(out, "comparison.csv")).read().splitlines()
    check("run_all: every method folder is scored with the same JSON (folders without videos ignored), a video a "
          "method was not run on is left out (not an error), and the methods are compared with their video counts",
          code == 0 and table[0] == "method,videos,gen_time_s"
          and table[1:] == ["method_a,1,10.0000", "method_b,2,40.0000"]
          and json.load(open(os.path.join(out, "method_a", "run_info.json")))["not_generated"] == ["p2"]
          and os.path.isfile(os.path.join(out, "comparison.md")), log[-1500:])
    check("run_all: --methods picks folders and refuses unknown ones",
          RA.method_folders(root, "method_b") == ["method_b"]
          and raises(lambda: RA.method_folders(root, "method_c"), SystemExit))
    video = os.path.join(tmp, "gen", "out.mp4")
    os.makedirs(os.path.dirname(video))
    make_cmd = [sys.executable, "-c", f"import shutil; shutil.copy({os.path.join(root, 'method_a', 'p1.mp4')!r}, "
                                      f"{video!r})"]
    code, _ = run_quiet(TR.main, ["--video", video, "--switch_latents", "14", "--"] + make_cmd)
    side = V.read_sidecar(video)
    check("timed_run: times the command and writes gen_time_s and switch frame 4*14-3 = 53 beside the video",
          code == 0 and side["switch_frames"] == [53] and side["gen_time_s"] >= 0 and "shutil" in side["command"])
    bad = os.path.join(tmp, "gen", "never.mp4")
    code, _ = run_quiet(TR.main, ["--video", bad, "--", sys.executable, "-c", "raise SystemExit(3)"])
    check("timed_run: a failed command records nothing and passes its exit code on",
          code == 3 and not os.path.exists(V.sidecar_path(bad)))


# ---------------------------------------------------------------- generation launcher
def load_wan_parser(generate_py):
    """Wan's own _parse_args/_validate_args, loaded without torch (as Visual Relay's CLI test does)."""
    import ast
    import random
    from types import SimpleNamespace
    tree = ast.parse(open(generate_py, encoding="utf-8").read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_parse_args", "_validate_args")]

    def cfg(shift):
        return SimpleNamespace(sample_steps=40, sample_shift=shift, sample_guide_scale=5, frame_num=81,
                               vae_stride=(4, 8, 8))
    scope = dict(argparse=argparse, random=random, sys=sys,
                 WAN_CONFIGS={"t2v-A14B": cfg(12.0), "i2v-A14B": cfg(5.0)},
                 SIZE_CONFIGS={"832*480": (832, 480)},
                 SUPPORTED_SIZES={"t2v-A14B": ["832*480"], "i2v-A14B": ["832*480"]},
                 EXAMPLE_PROMPT={"t2v-A14B": {"prompt": "x"}, "i2v-A14B": {"prompt": "x", "image": "x.jpg"}},
                 str2bool=lambda v: str(v).lower() in ("true", "1", "yes"))
    exec(compile(ast.Module(body=fns, type_ignores=[]), generate_py, "exec"), scope)

    def parse(argv):
        from unittest.mock import patch
        with patch.object(sys, "argv", ["generate.py"] + list(argv)):
            return scope["_parse_args"]()
    return parse


def load_h3_parser(generate_py):
    import ast
    from pathlib import Path
    tree = ast.parse(open(generate_py, encoding="utf-8").read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("str_to_bool", "parse_args")]
    scope = dict(argparse=argparse, Path=Path, json=json)
    exec(compile(ast.Module(body=fns, type_ignores=[]), generate_py, "exec"), scope)
    return scope["parse_args"]


def suite_for_launcher(tmp):
    imgs = []
    for k in (1, 2):
        path = os.path.join(tmp, f"v_img{k}.png")
        from PIL import Image
        Image.fromarray(np.full((48, 64, 3), 40 * k, np.uint8)).save(path)
        imgs.append(path)
    return write_json(os.path.join(tmp, "gen_suite.json"), [
        {"id": "p1", "file_name": "p1.mp4", "global_prompt": "A man on a court.",
         "local_prompts": ["He jumps.", "He runs \"fast\" toward the camera."]},
        {"id": "p2", "file_name": "p2.mp4", "global_prompt": "A dog in a park."},
        {"id": "v1", "file_name": "v1.mp4", "global_prompt": "A courtyard at night.",
         "local_prompts": ["The camera circles.", "The orbit ends."], "images": imgs, "image_latents": [0, 14]},
        {"id": "v3", "file_name": "v3.mp4", "global_prompt": "Three shots.",
         "local_prompts": ["a", "b", "c"], "images": imgs[:1]},
    ])


def test_generate_plan(tmp, wan_src, vr_src, h3_src):
    cfg = GS.load_config(os.path.join(ROOT, "methods.json"))
    entries = S.load_suite(suite_for_launcher(tmp), videos_dir=os.path.join(tmp, "o"))
    by = {e["id"]: e for e in entries}
    enabled = [m for m, spec in cfg["methods"].items() if spec.get("enabled", True)]
    jobs, skipped = GS.plan(entries, cfg, enabled, os.path.join(tmp, "o"), ["0", "1"], 29500)
    pairs = sorted((j["method"], j["id"]) for j in jobs)
    check("launcher: text methods run on text entries, plain I2V on any image entry, Visual Relay only on "
          "2-image/2-prompt entries; H3 is disabled in methods.json",
          pairs == [("wan2.2_i2v", "v1"), ("wan2.2_i2v", "v3"), ("wan2.2_i2v_visual_relay", "v1"),
                    ("wan2.2_t2v", "p1"), ("wan2.2_t2v", "p2"), ("wan2.2_t2v_prompt_relay", "p1"),
                    ("wan2.2_t2v_prompt_relay", "p2")]
          and set(enabled) == {"wan2.2_t2v", "wan2.2_t2v_prompt_relay", "wan2.2_i2v", "wan2.2_i2v_visual_relay"}, str(pairs))
    j = {(x["method"], x["id"]): x for x in jobs}
    t2v = j[("wan2.2_t2v", "p1")]["command"]
    vr = j[("wan2.2_i2v_visual_relay", "v1")]["command"]
    check("launcher: same file name in every method's folder; plain models get global + local prompts joined; "
          "Img2's latent 14 is recorded (switch frame 53) for both image methods; 2 GPUs = torchrun + FSDP + "
          "Ulysses 2",
          j[("wan2.2_t2v", "p1")]["output"].endswith(os.path.join("o", "wan2.2_t2v", "p1.mp4"))
          and t2v[t2v.index("--prompt") + 1] == 'A man on a court. He jumps. He runs "fast" toward the camera.'
          and vr[vr.index("--switch_latents") + 1] == "14"
          and "--switch_latents" in j[("wan2.2_i2v", "v1")]["command"]
          and "--switch_latents" not in j[("wan2.2_i2v", "v3")]["command"]
          and "torch.distributed.run" in t2v and t2v[-4:] == ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", "2"]
          and j[("wan2.2_t2v", "p1")]["gpus"] == "0,1", " ".join(t2v))
    one, _ = GS.plan([by["p1"]], cfg, ["wan2.2_t2v"], os.path.join(tmp, "o"), ["3"], 29503)
    c1 = one[0]["command"]
    check("launcher: 1 GPU = plain python with Wan's offloading, no FSDP; 3 GPUs refused (Ulysses needs 1/2/4/8)",
          "torch.distributed.run" not in c1 and c1[-3:] == ["--offload_model", "True", "--convert_model_dtype"]
          and "--dit_fsdp" not in c1 and one[0]["gpus"] == "3"
          and raises(lambda: GS.plan([by["p1"]], cfg, ["wan2.2_t2v"], tmp, ["0", "1", "2"], 1), GS.ConfigError))
    bad = dict(by["v1"], image_latents=None)
    check("launcher: an image entry without image_latents is refused for Visual Relay; a latent past the "
          "video's end is refused",
          raises(lambda: GS.plan([bad], cfg, ["wan2.2_i2v_visual_relay"], tmp, ["0"], 1), GS.ConfigError)
          and raises(lambda: GS.plan([dict(by["v1"], image_latents=[0, 21])], cfg, ["wan2.2_i2v_visual_relay"], tmp, ["0"], 1),
                     GS.ConfigError))

    if wan_src and vr_src:
        parse_wan, parse_vr = load_wan_parser(wan_src), load_wan_parser(vr_src)
        ok, notes = True, []
        for gpus in (["0", "1"], ["0"]):
            js, _ = GS.plan(entries, cfg, enabled, os.path.join(tmp, "o"), gpus, 29500)
            for job in js:
                if job["uses_prompt_file"]:
                    GS.write_prompt_file(job, by[job["id"]])
                cmd = job["command"]
                argv = cmd[cmd.index("generate.py") + 1:]
                try:
                    a = (parse_vr if job["method"] == "wan2.2_i2v_visual_relay" else parse_wan)(argv)
                except BaseException as err:  # noqa: BLE001
                    ok = False
                    notes.append(f"{job['method']}/{job['id']}: {type(err).__name__} {err}")
                    continue
                relay_text = 'A man on a court.He jumps. He runs "fast" toward the camera.'  # the fork's own join
                if job["method"] == "wan2.2_t2v_prompt_relay" and job["id"] == "p1" and a.prompt != relay_text:
                    ok = False
                    notes.append(f"relay prompt {a.prompt!r}")
                if job["method"] == "wan2.2_i2v_visual_relay" and (a.image_latent_index != [0, 14] or len(a.image) != 2
                                                        or a.sample_shift != 3.0 or a.base_seed != 48216):
                    ok = False
                    notes.append(f"visual relay args {vars(a)}")
        check("launcher contract: every Wan command is accepted by the code it runs (Prompt Relay fork's and "
              "Visual Relay's own argument checks, 1 and 2 GPUs); Visual Relay gets Img2 at latent 14, shift 3, "
              "seed 48216", ok, "; ".join(notes))
    else:
        print("[SKIP] launcher contract with the Wan code (needs src/Wan2.2 and src/Visual-Relay, or "
              "--wan_src / --visual_relay_src)")
    if h3_src:
        parse_h3 = load_h3_parser(h3_src)
        cfg_h3 = json.loads(json.dumps(cfg))
        for m in ("minimax_h3", "minimax_h3_prompt_relay"):
            cfg_h3["methods"][m]["enabled"] = True
        js, _ = GS.plan(entries, cfg_h3, ["minimax_h3", "minimax_h3_prompt_relay"], os.path.join(tmp, "o"), ["0"], 1)
        parsed = []
        for job in js:
            cmd = job["command"]
            parsed.append(parse_h3(cmd[cmd.index("generate.py") + 1:]))
        check("launcher contract: the (disabled) H3 placeholders are accepted by H3's own argument parser",
              len(parsed) == 4 and all(a.frame_num == 124 and a.seed == 48216 for a in parsed)
              and sum(a.prompt_filepath is not None for a in parsed) == 2)
    else:
        print("[SKIP] launcher contract with the H3 code (needs src/MiniMax-H3 or --h3_src)")


def test_generate_run(tmp):
    """End to end with a stand-in generator that copies a small video to the requested path."""
    sample = os.path.join(tmp, "gen_sample.mp4")
    make_video(sample, n=81)
    code_dir = os.path.join(tmp, "fake_model")
    os.makedirs(code_dir)
    with open(os.path.join(code_dir, "fake_generate.py"), "w", encoding="utf-8") as f:
        f.write("import shutil, sys\n"
                "a = sys.argv[1:]\n"
                "out = a[a.index('--save_file') + 1]\n"
                "if 'FAIL' in ' '.join(a): sys.exit(5)\n"
                f"shutil.copy({sample!r}, out)\n")
    ckpt = os.path.join(tmp, "ckpt")
    os.makedirs(ckpt)
    base = {"runs_on": "text", "python": sys.executable, "code_dir": code_dir, "script": "fake_generate.py",
            "parallel": "none"}
    cfg = {"paths": {"py": sys.executable, "my_ckpt": ckpt}, "settings": {"seed": 1, "wan_frame_num": 81},
           "methods": {
               "plain": dict(base, args=["--ckpt", "{my_ckpt}", "--prompt", "{prompt_joined}",
                                         "--save_file", "{output}"]),
               "relay": dict(base, args=["--prompt_filepath", "{prompt_file}", "--save_file", "{output}"]),
               "img": dict(base, runs_on="two_images", record_switch="image_latents",
                           args=["--image", "{image_1}", "--image", "{image_2}", "--save_file", "{output}"]),
               "off": dict(base, enabled=False, args=["--save_file", "{output}"]),
           }}
    cfg_path = write_json(os.path.join(tmp, "methods_test.json"), cfg)
    suite = suite_for_launcher(tmp)
    out = os.path.join(tmp, "gen_out")
    code, log = run_quiet(GS.main, ["--prompts", suite, "--config", cfg_path, "--out_root", out, "--gpus", "0"])
    side = V.read_sidecar(os.path.join(out, "img", "v1.mp4"))
    relay_in = json.load(open(os.path.join(out, "_prompt_files", "p1.json")))
    made = sorted(f"{m}/{f}" for m in os.listdir(out) if os.path.isdir(os.path.join(out, m))
                  for f in os.listdir(os.path.join(out, m)) if f.endswith(".mp4"))
    check("launcher run: every applicable video is made under OUT/<method>/<file_name> with its generation time; "
          "Visual-Relay-style methods record switch frame 53; Prompt Relay gets a prompt file; disabled methods "
          "make nothing",
          code == 0 and made == ["img/v1.mp4", "plain/p1.mp4", "plain/p2.mp4", "relay/p1.mp4", "relay/p2.mp4"]
          and side.get("switch_frames") == [53] and side.get("gen_time_s") is not None
          and V.read_sidecar(os.path.join(out, "plain", "p1.mp4")).get("switch_frames") is None
          and relay_in == {"global_prompt": "A man on a court.",
                           "local_prompts": ["He jumps.", 'He runs "fast" toward the camera.']}
          and not os.path.exists(os.path.join(out, "off"))
          and os.path.isfile(os.path.join(out, "plain", "logs", "p1.log")), f"{made} {side} {log[-800:]}")
    code2, log2 = run_quiet(GS.main, ["--prompts", suite, "--config", cfg_path, "--out_root", out, "--gpus", "0"])
    check("launcher run: a second run keeps existing videos", code2 == 0 and log2.count("exists, kept") == 5, log2)

    cfg["methods"]["plain"]["args"] = ["--prompt", "FAIL", "--save_file", "{output}"]
    cfg_fail = write_json(os.path.join(tmp, "methods_fail.json"), cfg)
    code3, log3 = run_quiet(GS.main, ["--prompts", suite, "--config", cfg_fail, "--out_root", out, "--gpus", "0",
                                      "--methods", "plain", "--ids", "p2", "--force"])
    check("launcher run: a failed generation is reported with its log and exit code 1; --force deletes the old "
          "video first", code3 == 1 and "FAILED" in log3
          and not os.path.isfile(os.path.join(out, "plain", "p2.mp4")), log3)

    cfg["methods"]["plain"]["args"] = ["--ckpt", "{my_ckpt}", "--save_file", "{output}"]
    cfg["paths"]["my_ckpt"] = os.path.join(tmp, "no_such_ckpt")
    cfg_missing = write_json(os.path.join(tmp, "methods_missing.json"), cfg)
    code4, log4 = run_quiet(GS.main, ["--prompts", suite, "--config", cfg_missing, "--out_root", out, "--gpus", "0",
                                      "--methods", "plain", "--force"])
    code5, log5 = run_quiet(GS.main, ["--prompts", suite, "--config", cfg_path, "--out_root",
                                      os.path.join(tmp, "printed"), "--gpus", "0", "--print_only"])
    check("launcher run: a missing checkpoint folder stops the method before anything runs; --print_only writes "
          "nothing", code4 == 1 and "checkpoint folder not found" in log4
          and code5 == 0 and "fake_generate.py" in log5 and not os.path.exists(os.path.join(tmp, "printed")),
          log4 + log5)

    ev = os.path.join(tmp, "gen_eval")
    code6, _ = run_quiet(RA.main, ["--prompts", suite, "--root", out, "--out", ev, "--only", "gen_time"])
    table = open(os.path.join(ev, "comparison.csv")).read().splitlines()
    check("launcher + run_all: the generated folders score straight away (each method over the entries it ran on)",
          code6 == 0 and [r.split(",")[:2] for r in table[1:]] == [["img", "1"], ["plain", "1"], ["relay", "2"]],
          str(table))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbench_root", default=MV.default_vbench_root())
    repo = os.path.dirname(ROOT)

    def found(path):
        return path if os.path.isfile(path) else None
    ap.add_argument("--wan_src", default=found(os.path.join(repo, "src", "Wan2.2", "generate.py")),
                    help="Prompt Relay's Wan2.2/generate.py (default: the submodule)")
    ap.add_argument("--visual_relay_src", default=found(os.path.join(repo, "src", "Visual-Relay", "Wan2.2",
                                                                     "generate.py")))
    ap.add_argument("--h3_src", default=found(os.path.join(repo, "src", "MiniMax-H3", "generate.py")))
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="vbe_test_")
    try:
        test_suite(tmp)
        test_segments()
        test_jobs()
        if a.vbench_root:
            test_vbench_contract(tmp, a.vbench_root)
        else:
            print("[SKIP] VBench contract checks (pass --vbench_root PATH_TO_VBENCH_CLONE)")
        test_extra_math()
        test_report(tmp)
        test_end_to_end(tmp)
        test_run_all_and_timer(tmp)
        test_generate_plan(tmp, a.wan_src, a.visual_relay_src, a.h3_src)
        test_generate_run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    passed = sum(RESULTS)
    print(f"\n{passed} of {len(RESULTS)} checks passed" + ("" if passed == len(RESULTS) else " - SOME FAILED"))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
