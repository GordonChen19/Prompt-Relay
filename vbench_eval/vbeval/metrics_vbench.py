"""VBench and VBench++ (I2V) dimensions, run through VBench's own code.

VBench scores whole files and knows nothing about prompts changing over time, so this module hands it
short clips (segments, segment cores, switch windows) as separate "videos", each with its own prompt or
input image, and maps the scores back. Each dimension's model is loaded once for all clips.

Calls VBench as its own VBench.evaluate() does: init_submodules([dim]) then compute_<dim>(full_info_json,
device, submodules). Written against Vchitect/VBench commit fd18b3d (2026-08-21).
"""
import importlib
import json
import os
import sys

VBENCH_DIMS = [
    "overall_consistency",     # text-video alignment (ViCLIP), per segment with its local prompt
    "subject_consistency",     # DINO, frame-to-frame and to the clip's first frame
    "background_consistency",  # CLIP, same
    "temporal_flickering",     # mean absolute frame difference
    "motion_smoothness",       # AMT frame interpolation error
    "dynamic_degree",          # RAFT optical flow: does it move (0/1)
    "aesthetic_quality",       # LAION aesthetic predictor
    "imaging_quality",         # MUSIQ (over-exposure, blur, noise)
]
I2V_DIMS = [
    "i2v_subject",             # VBench++: DINO similarity to an input image
    "i2v_background",          # VBench++: DreamSim similarity to an input image
]
ALL_DIMS = VBENCH_DIMS + I2V_DIMS

# Which clips each dimension scores (see README "How a video is cut").
WHOLE = {"overall_consistency", "subject_consistency", "background_consistency", "temporal_flickering",
         "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"}
FULL_SEGMENTS = {"subject_consistency", "background_consistency", "dynamic_degree"}
CORE_SEGMENTS = {"overall_consistency"}
SWITCH_WINDOWS = {"subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness"}
MIN_WINDOW_FRAMES = 8   # shorter switch windows are not scored by VBench (AMT and DINO need a few frames)

# VBench reports imaging quality per video on a 0-100 scale (it divides only the average by 100).
SCALE = {"imaging_quality": 0.01}


BUNDLED_VBENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "VBench")


def default_vbench_root():
    """$VBENCH_ROOT if set, else the VBench submodule at vbench_eval/third_party/VBench if it is checked out."""
    env = os.environ.get("VBENCH_ROOT")
    if env:
        return env
    return BUNDLED_VBENCH if os.path.isdir(os.path.join(BUNDLED_VBENCH, "vbench")) else None


def import_vbench(vbench_root=None):
    """Make 'vbench' (and 'vbench2_beta_i2v') importable from a git clone if it is not pip-installed."""
    if vbench_root:
        root = os.path.abspath(vbench_root)
        if not os.path.isdir(os.path.join(root, "vbench")):
            raise FileNotFoundError(f"--vbench_root {root} has no 'vbench' folder (expected a VBench git clone)")
        if root not in sys.path:
            sys.path.insert(0, root)
    importlib.import_module("vbench")


def clips_needed(video, dims):
    """The clip files a video needs for the given dimensions: {'core_seg1': (start, end), ...}."""
    need = {}
    n = len(video["segments"])
    for s in video["segments"]:
        k = s["index"]
        if dims & CORE_SEGMENTS and video["seg_texts"]:
            need[f"core_seg{k}"] = s["core"]
        if n > 1 and (dims & FULL_SEGMENTS or (dims & set(I2V_DIMS) and video["images"][k - 1])):
            need[f"full_seg{k}"] = s["full"]
    for w in video["switches"]:
        lo, hi = w["window"]
        if dims & SWITCH_WINDOWS and hi - lo >= MIN_WINDOW_FRAMES:
            need[f"switch{w['index']}"] = w["window"]
    return need


def build_jobs(video, dim):
    """[(slot, clip path, prompt, image)] for one video and one dimension."""
    jobs = []
    n = len(video["segments"])
    clips = video["clips"]
    if dim == "overall_consistency":
        jobs.append(("global", video["path"], video["whole_text"], None))
        for k, text in enumerate(video["seg_texts"], start=1):
            jobs.append((f"seg{k}", clips[f"core_seg{k}"], text, None))
        return jobs
    if dim in I2V_DIMS:
        for k, image in enumerate(video["images"], start=1):
            if image:
                clip = clips[f"full_seg{k}"] if n > 1 else video["path"]
                jobs.append((f"img{k}", clip, video["whole_text"], image))
        return jobs
    if dim in WHOLE:
        jobs.append(("whole", video["path"], video["whole_text"], None))
    if dim in FULL_SEGMENTS and n > 1:
        for k in range(1, n + 1):
            jobs.append((f"seg{k}", clips[f"full_seg{k}"], video["whole_text"], None))
    if dim in SWITCH_WINDOWS:
        for w in video["switches"]:
            key = f"switch{w['index']}"
            if key in clips:
                jobs.append((key, clips[key], video["whole_text"], None))
    return jobs


def write_full_info(dim, jobs, path):
    """The list VBench's load_dimension_info / load_i2v_dimension_info read: one entry per clip."""
    info = []
    for _video_id, _slot, clip, prompt, image in jobs:
        entry = {"prompt_en": prompt or "", "dimension": [dim], "video_list": [clip]}
        if dim in I2V_DIMS:
            entry["custom_image_path"] = image
        info.append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


def run_dimension(dim, jobs, work_dir, device, local=False):
    """Score every clip in jobs [(video_id, slot, clip, prompt, image)] -> {(video_id, slot): score}."""
    if not jobs:
        return {}
    clips = [os.path.abspath(j[2]) for j in jobs]
    if len(set(clips)) != len(clips):
        raise RuntimeError(f"{dim}: the same clip appears twice; scores could not be told apart")
    info_path = write_full_info(dim, jobs, os.path.join(work_dir, f"{dim}_full_info.json"))
    if dim in I2V_DIMS:
        utils = importlib.import_module("vbench2_beta_i2v.utils")
        module = importlib.import_module(f"vbench2_beta_i2v.{dim}")
    else:
        utils = importlib.import_module("vbench.utils")
        module = importlib.import_module(f"vbench.{dim}")
    submodules = utils.init_submodules([dim], local=local, read_frame=False)[dim]
    _average, video_results = getattr(module, f"compute_{dim}")(info_path, device, submodules)
    by_clip = {}
    for r in video_results:
        by_clip[os.path.abspath(r["video_path"])] = float(r["video_results"]) * SCALE.get(dim, 1.0)
    return {(j[0], j[1]): by_clip.get(os.path.abspath(j[2]), float("nan")) for j in jobs}
