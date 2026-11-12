"""Turning per-clip scores into one row per video, and writing the result files."""
import csv
import json
import math
import os

# Column order for the CSV files and the printed table (anything else follows, sorted).
ORDER = [
    "overall_consistency_seg_mean", "clip_t_seg_mean", "prompt_match_acc",
    "overall_consistency_global", "clip_t_global",
    "subject_consistency", "subject_consistency_seg_mean", "subject_consistency_switch_min",
    "background_consistency", "background_consistency_seg_mean", "background_consistency_switch_min",
    "clip_f_mean", "clip_f_min", "clip_f_switch_min",
    "temporal_flickering", "temporal_flickering_switch_min",
    "motion_smoothness", "motion_smoothness_switch_min",
    "dynamic_degree", "dynamic_degree_seg_mean",
    "aesthetic_quality", "imaging_quality",
    "i2v_subject_img1", "i2v_subject_img2", "i2v_background_img1", "i2v_background_img2",
    "gen_time_s",
]


def finite(values):
    return [float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]


def mean(values):
    vals = finite(values)
    return sum(vals) / len(vals) if vals else float("nan")


def minimum(values):
    vals = finite(values)
    return min(vals) if vals else float("nan")


def _slots(scores, prefix):
    """Values of slots 'seg1', 'seg2', ... (or 'switch1', ...) in slot order."""
    keys = sorted((k for k in scores if k.startswith(prefix) and k[len(prefix):].isdigit()),
                  key=lambda k: int(k[len(prefix):]))
    return [(k, scores[k]) for k in keys]


def flatten(vbench_scores, extra_scores):
    """One flat {metric: value} dict for a video.
    vbench_scores: {dimension: {slot: score}} with slots whole / global / segK / switchK / imgK.
    extra_scores: already-flat values from metrics_extra (and gen_time_s)."""
    m = {}
    oc = vbench_scores.get("overall_consistency")
    if oc is not None:
        m["overall_consistency_global"] = oc.get("global", float("nan"))
        segs = _slots(oc, "seg")
        for k, v in segs:
            m[f"overall_consistency_{k}"] = v
        if segs:
            m["overall_consistency_seg_mean"] = mean(v for _, v in segs)
    for dim in ("subject_consistency", "background_consistency", "dynamic_degree"):
        d = vbench_scores.get(dim)
        if d is None:
            continue
        m[dim] = d.get("whole", float("nan"))
        segs = _slots(d, "seg")
        if segs:
            m[f"{dim}_seg_mean"] = mean(v for _, v in segs)
            for k, v in segs:
                m[f"{dim}_{k}"] = v
    for dim in ("subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness"):
        d = vbench_scores.get(dim)
        if d is None:
            continue
        m.setdefault(dim, d.get("whole", float("nan")))
        switches = _slots(d, "switch")
        if switches:
            m[f"{dim}_switch_min"] = minimum(v for _, v in switches)
    for dim in ("aesthetic_quality", "imaging_quality"):
        if dim in vbench_scores:
            m[dim] = vbench_scores[dim].get("whole", float("nan"))
    for dim in ("i2v_subject", "i2v_background"):
        for k, v in _slots(vbench_scores.get(dim, {}), "img"):
            m[f"{dim}_{k}"] = v
    m.update({k: v for k, v in extra_scores.items() if not isinstance(v, (list, dict))})
    return m


def summarize(records):
    """Per metric over videos: mean, standard deviation and count of finite values."""
    keys = []
    for r in records:
        for k in r.get("metrics", {}):
            if k not in keys:
                keys.append(k)
    out = {}
    for k in ordered(keys):
        vals = finite(r.get("metrics", {}).get(k) for r in records)
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        out[k] = {"mean": mu, "std": sd, "n": len(vals)}
    return out


def ordered(keys):
    first = [k for k in ORDER if k in keys]
    seg_like = sorted(k for k in keys if k not in ORDER)
    return first + seg_like


def _clean(x):
    """NaN -> None so the JSON files are strict JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2, ensure_ascii=False)


def load_records(out_dir):
    path = os.path.join(out_dir, "per_video.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}


def write_outputs(out_dir, records, summary, run_info):
    records = sorted(records, key=lambda r: r["id"])
    write_json(os.path.join(out_dir, "per_video.json"), records)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_json(os.path.join(out_dir, "run_info.json"), run_info)
    keys = ordered({k for r in records for k in r.get("metrics", {})})
    with open(os.path.join(out_dir, "per_video.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "video", "frames", "fps", "switch_frames", "errors"] + keys)
        for r in records:
            row = [r["id"], r.get("video", ""), r.get("n_frames", ""), r.get("fps", ""),
                   " ".join(str(c) for c in r.get("switch_frames", []) or []),
                   "; ".join(f"{k}: {v}" for k, v in (r.get("errors") or {}).items())]
            row += [_cell(r.get("metrics", {}).get(k)) for k in keys]
            w.writerow(row)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n"])
        for k, s in summary.items():
            w.writerow([k, _cell(s["mean"]), _cell(s["std"]), s["n"]])


def _cell(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.6g}" if isinstance(v, float) else v


def format_summary(summary, title):
    lines = [title, "  %-36s %10s %10s %5s" % ("metric", "mean", "std", "n")]
    for k, s in summary.items():
        lines.append("  %-36s %10s %10s %5d" % (k, _cell(s["mean"]) or "-", _cell(s["std"]) or "-", s["n"]))
    return "\n".join(lines)
