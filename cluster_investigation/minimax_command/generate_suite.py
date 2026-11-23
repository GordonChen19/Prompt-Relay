#!/usr/bin/env python
"""Generate every method's video for every prompt in the suite JSON, ready for run_all.py, on all given GPUs.

    python vbench_eval/generate_suite.py --prompts PROMPTS.json [--methods a,b] [--ids p001,v001]
                                         [--gpus 0,1,2,3] [--gpus_per_job 1] [--batch_size 8]
                                         [--out_root outputs] [--config vbench_eval/methods.json]
                                         [--print_only] [--status] [--force]

Each method in methods.json says which code, Python environment, checkpoint and arguments it uses, which
entries it applies to, and on which timeline its events sit (so every video gets the frames of each local
prompt recorded next to it). Videos go to OUT_ROOT/<method>/<file_name>, the same file name for every
method, with a <name>.meta.json beside each and the generation logs in OUT_ROOT/<method>/logs/.

GPUs: the listed GPUs are split into groups of --gpus_per_job; each group runs one job at a time and picks
up the next waiting job as soon as it is free, across all the methods asked for. Methods marked "batch"
load their model once and make --batch_size videos in a row. Methods with "parallel": "wan" split one model
over the whole group (sharding: FSDP + Ulysses, no offloading), so --gpus 0,1 --gpus_per_job 2 makes each
video on both GPUs; every other method uses the first GPU of its group.
Videos that already exist are kept (--force regenerates). Methods with "enabled": false are skipped.
--status shows how many videos each method has made so far (safe to run while a generation is going).
Run from anywhere; any Python 3 works (this script only starts the methods' own environments).
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DRIVERS = os.path.join(HERE, "drivers")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vbeval import suite as S  # noqa: E402
from vbeval import timing as T  # noqa: E402

RUNS_ON = ("text", "first_image", "images", "two_images")
PARALLEL = ("wan", "none")
TIMELINE = ("wan", "h3", "image_latents", "own")
WAN_GPU_COUNTS = (1, 2, 4, 8)   # Ulysses splits Wan's 40 attention heads evenly


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- configuration
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("paths", "settings", "methods"):
        if not isinstance(cfg.get(key), dict):
            raise ConfigError(f"{path}: needs a '{key}' object")
    for name, m in cfg["methods"].items():
        for key in ("runs_on", "python", "code_dir", "script", "parallel", "timeline", "args"):
            if key not in m:
                raise ConfigError(f"method {name}: missing '{key}'")
        if m["runs_on"] not in RUNS_ON:
            raise ConfigError(f"method {name}: runs_on must be one of {RUNS_ON}")
        if m["parallel"] not in PARALLEL:
            raise ConfigError(f"method {name}: parallel must be one of {PARALLEL}")
        if m["timeline"] not in TIMELINE:
            raise ConfigError(f"method {name}: timeline must be one of {TIMELINE}")
        if "record_switch" in m:
            raise ConfigError(f"method {name}: 'record_switch' is now 'timeline': 'image_latents'")
        if not isinstance(m["args"], list) or not all(isinstance(a, str) for a in m["args"]):
            raise ConfigError(f"method {name}: args must be a list of strings")
        if m.get("batch") and "{job_files}" not in m["args"]:
            raise ConfigError(f"method {name}: a batch method needs '{{job_files}}' in its args")
    return cfg


def resolve_path(p):
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def path_value(key, value):
    """'*_model' entries may be a Hugging Face name (kept as is) or a path; every other entry is a path."""
    value = str(value)
    if key.endswith("_model") and not value.startswith(("~", "/", "./", "../")):
        return value
    return resolve_path(value)


def fill(template, values, where):
    try:
        return template.format_map(values)
    except KeyError as err:
        raise ConfigError(f"{where}: unknown placeholder {{{err.args[0]}}} in {template!r}") from None


def frames_on(timeline, settings):
    if timeline == "h3":
        return int(settings.get("h3_frame_num", 124))
    return int(settings.get("wan_frame_num", 81))


# ---------------------------------------------------------------- what runs on what
def joined_prompt(entry):
    return " ".join(p.strip() for p in [entry["global_prompt"], *entry["local_prompts"]] if p and p.strip())


def applies(method, entry):
    """None if the method runs on this entry, else the reason it does not."""
    images = [i for i in entry["images"] if i]
    kind = method["runs_on"]
    if kind == "first_image" and not images:
        return "entry has no image"
    if kind == "images" and (len(images) < 2 or len(images) != len(entry["local_prompts"])):
        return "needs 2 or more images, one per local prompt"
    if kind == "two_images" and (len(images) != 2 or len(entry["local_prompts"]) != 2):
        return "needs exactly 2 images and 2 local prompts"
    return None                                   # 'text' runs on every entry; its images are not used


def image_latents_on_wan(entry, cfg):
    """The entry's image latents (given for an 81-frame Wan video) on Wan at the configured length."""
    return T.model_image_latents(S.image_latents(entry), "wan", frames_on("wan", cfg["settings"]))


def event_plan(name, method, cfg, entry):
    """What the job records about where events are: ('event_frames', [[s, e], ...]),
    ('switch_latents', [L2, L3, ...]) or (None, None) when the method's own code records it."""
    tl = method["timeline"]
    n_prompts = len(entry["local_prompts"])
    if tl in ("wan", "h3"):
        if n_prompts < 2:
            return None, None
        n_frames = frames_on(tl, cfg["settings"])
        try:
            frames = T.event_frames(entry["timing"], n_prompts, tl, n_frames)
        except T.TimingError as err:
            raise ConfigError(f"{entry['id']} / {name}: {err}") from None
        return "event_frames", [list(x) for x in frames]
    if tl == "image_latents":
        images = [i for i in entry["images"] if i]
        if len(images) < 2:
            return None, None
        latents = image_latents_on_wan(entry, cfg)
        n_lat = T.latent_count("wan", frames_on("wan", cfg["settings"]))
        for L in latents[1:]:
            if not 1 <= L < n_lat:
                raise ConfigError(f"{entry['id']}: image latent {L} is outside 1..{n_lat - 1}")
        return "switch_latents", latents[1:]
    return None, None


def prompt_relay_file(method, cfg, entry):
    """Prompt Relay's own JSON for this method: global and local prompts, plus the entry's timing moved
    onto this model's latents (left out when the entry gives none, so the model uses its default)."""
    data = {"global_prompt": entry["global_prompt"] or " ".join(entry["local_prompts"]),
            "local_prompts": entry["local_prompts"]}
    timing = entry["timing"] or {}
    for key in ("mode", "sigma"):
        if key in timing:
            data[key] = timing[key]
    tl = method["timeline"]
    if (timing.get("frame_intervals") or timing.get("segment_lengths")) and tl in ("wan", "h3"):
        n_frames = frames_on(tl, cfg["settings"])
        data["frame_intervals"] = [list(x) for x in T.model_latent_intervals(
            timing, len(entry["local_prompts"]), tl, n_frames)]
    return data


def job_record(name, method, cfg, entry, output):
    """Everything a driver needs to make one video (written as OUT/_jobs/<method>/<id>.json)."""
    kind, value = event_plan(name, method, cfg, entry)
    images = [os.path.abspath(i) for i in entry["images"] if i]
    tl = method["timeline"]
    rec = {"method": name, "id": entry["id"], "output": output, "seed": int(cfg["settings"].get("seed", 0)),
           "global_prompt": entry["global_prompt"], "local_prompts": entry["local_prompts"],
           "prompt_joined": joined_prompt(entry), "images": images,
           "image_latents": image_latents_on_wan(entry, cfg) if images else [],
           "subject": entry.get("subject"), "event_words": entry.get("event_words"),
           "timing": entry["timing"] or {}, "event_frames": value if kind == "event_frames" else None,
           "uses_images": method["runs_on"] != "text",
           "prompt_relay": prompt_relay_file(method, cfg, entry)}
    if images:
        rec["image_frames"] = [T.latent_to_frame("wan", L, frames_on("wan", cfg["settings"]))
                               for L in rec["image_latents"]]
    if kind == "switch_latents":
        rec["switch_frames"] = [T.latent_to_frame("wan", L, frames_on("wan", cfg["settings"])) for L in value]
    if tl in ("wan", "h3") and len(entry["local_prompts"]) >= 1:
        rec["latent_intervals"] = [list(x) for x in T.model_latent_intervals(
            entry["timing"], len(entry["local_prompts"]), tl, frames_on(tl, cfg["settings"]))]
    return rec


def entry_values(entry, output, prompt_file, job_file, cfg):
    values = {"output": output, "prompt_file": prompt_file, "job_file": job_file,
              "prompt_joined": joined_prompt(entry), "global_prompt": entry["global_prompt"],
              "drivers": DRIVERS}
    for k, text in enumerate(entry["local_prompts"], start=1):
        values[f"local_{k}"] = text
    images = [i for i in entry["images"] if i]
    for k, image in enumerate(images, start=1):
        values[f"image_{k}"] = os.path.abspath(image)
    for k, latent in enumerate(image_latents_on_wan(entry, cfg) if images else [], start=1):
        values[f"image_latent_{k}"] = str(latent)
    return values


IMAGE_ARG_TEMPLATE = ["--image", "{image}", "--image_latent_index", "{latent}", "--image_prompt", "{local}"]


def image_args(method, entry, cfg, values, where):
    """'{image_args}' in a method's args: one group per image (default: --image P --image_latent_index L
    --image_prompt TEXT, as Visual Relay's CLI takes them), so a method can take any number of images."""
    template = method.get("image_arg_template", IMAGE_ARG_TEMPLATE)
    images = [os.path.abspath(i) for i in entry["images"] if i]
    latents = image_latents_on_wan(entry, cfg)
    out = []
    for k, (img, lat) in enumerate(zip(images, latents)):
        local = entry["local_prompts"][k] if k < len(entry["local_prompts"]) else ""
        out.extend(fill(t, {**values, "image": img, "latent": str(lat), "local": local}, where) for t in template)
    return out


def base_values(cfg):
    return {**{k: path_value(k, v) for k, v in cfg["paths"].items()},
            **{k: str(v) for k, v in cfg["settings"].items()}, "drivers": DRIVERS}


def build_job(name, method, cfg, entry, out_root):
    """One video: its files, and (for per-video methods) the command template filled for this entry."""
    out_dir = os.path.join(out_root, name)
    output = os.path.join(out_dir, entry["file_name"])
    prompt_file = os.path.join(out_root, "_prompt_files", name, f"{entry['id']}.json")
    job_file = os.path.join(out_root, "_jobs", name, f"{entry['id']}.json")
    values = {**base_values(cfg), **entry_values(entry, output, prompt_file, job_file, cfg)}
    where = f"method {name}"
    kind, value = event_plan(name, method, cfg, entry)
    job = {"method": name, "id": entry["id"], "output": output, "prompt_file": prompt_file, "job_file": job_file,
           "uses_prompt_file": any("{prompt_file}" in a for a in method["args"]),
           "record": job_record(name, method, cfg, entry, output),
           "prompt_relay": prompt_relay_file(method, cfg, entry),
           "events": (kind, value), "log": os.path.join(out_dir, "logs", f"{entry['id']}.log")}
    if not method.get("batch"):
        job["args"] = []
        for a in method["args"]:
            if a == "{image_args}":
                job["args"].extend(image_args(method, entry, cfg, values, where))
            else:
                job["args"].append(fill(a, values, where))
    return job


def method_runtime(name, method, cfg):
    values = base_values(cfg)
    where = f"method {name}"
    python = resolve_path(fill(method["python"], values, where))
    code_dir = resolve_path(fill(method["code_dir"], values, where))
    script = fill(method["script"], values, where)
    script = script if os.path.isabs(script) else os.path.join(code_dir, script)
    return python, code_dir, script


def plan(entries, cfg, methods, out_root):
    jobs, skipped = [], []
    for name in methods:
        method = cfg["methods"][name]
        for entry in entries:
            reason = applies(method, entry)
            if reason:
                skipped.append((name, entry["id"], reason))
                continue
            jobs.append(build_job(name, method, cfg, entry, out_root))
    return jobs, skipped


# ---------------------------------------------------------------- tasks (what one GPU group runs)
def make_tasks(jobs, cfg, batch_size, interleave=False):
    """Per-video methods: one task per job. Batch methods: their jobs in groups of batch_size."""
    tasks, by_method = [], {}
    for job in jobs:
        by_method.setdefault(job["method"], []).append(job)
    for name, mjobs in by_method.items():
        method = cfg["methods"][name]
        if method.get("batch"):
            size = max(1, int(method.get("batch_size", batch_size)))
            for i in range(0, len(mjobs), size):
                chunk = mjobs[i:i + size]
                tasks.append({"method": name, "jobs": chunk, "batch": True,
                              "log": os.path.join(os.path.dirname(chunk[0]["log"]),
                                                  f"batch_{i // size + 1:03d}_{chunk[0]['id']}.log")})
        else:
            tasks.extend({"method": name, "jobs": [j], "batch": False, "log": j["log"]} for j in mjobs)
    order = list(by_method)
    tasks.sort(key=lambda t: order.index(t["method"]))
    if interleave:                       # method A batch 1, method B batch 1, method A batch 2, ...
        queues = [[t for t in tasks if t["method"] == m] for m in order]
        tasks = [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]
    return tasks


def task_command(task, cfg, group, master_port):
    """(command, cwd, CUDA_VISIBLE_DEVICES) for a task on a GPU group."""
    name = task["method"]
    method = cfg["methods"][name]
    python, code_dir, script = method_runtime(name, method, cfg)
    where = f"method {name}"
    if task["batch"]:
        values = base_values(cfg)
        args = []
        for a in method["args"]:
            if a == "{job_files}":
                args.extend(j["job_file"] for j in task["jobs"])
            else:
                args.append(fill(a, values, where))
        if method["parallel"] == "wan" and len(group) > 1:     # one model sharded over the group
            n = len(group)
            if n not in WAN_GPU_COUNTS:
                raise ConfigError(f"Wan runs on {WAN_GPU_COUNTS} GPUs, got {n} ({','.join(group)})")
            cmd = [python, "-m", "torch.distributed.run", f"--nproc_per_node={n}", f"--master_port={master_port}",
                   script, *args, "--ulysses_size", str(n), "--offload_model", "False"]
            return cmd, code_dir, ",".join(group)
        return [python, script, *args], code_dir, group[0]
    job = task["jobs"][0]
    args = job["args"]
    if method["parallel"] == "wan":
        n = len(group)
        if n not in WAN_GPU_COUNTS:
            raise ConfigError(f"Wan runs on {WAN_GPU_COUNTS} GPUs, got {n} ({','.join(group)})")
        offload = str(cfg["settings"].get("wan_offload_model", "True"))
        if n == 1:
            cmd = [python, script, *args, "--offload_model", offload, "--convert_model_dtype"]
        else:
            cmd = [python, "-m", "torch.distributed.run", f"--nproc_per_node={n}", f"--master_port={master_port}",
                   script, *args, "--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(n)]
        visible = ",".join(group)
    else:
        cmd = [python, script, *args]
        visible = group[0]
    wrapper = [sys.executable, os.path.join(HERE, "timed_run.py"), "--video", job["output"]]
    kind, value = job["events"]
    if method["runs_on"] != "text":
        wrapper.append("--uses_images")
    if kind == "event_frames":
        wrapper += ["--event_frames", *[f"{s}:{e}" for s, e in value]]
    elif kind == "switch_latents":
        wrapper += ["--switch_latents", *[str(v) for v in value]]
    return wrapper + ["--"] + cmd, code_dir, visible


def missing_files(name, method, cfg):
    """Things that must exist before a method can run (checked once per method)."""
    python, code_dir, script = method_runtime(name, method, cfg)
    problems = []
    if not os.path.isfile(python):
        problems.append(f"Python not found: {python} (set 'paths' in methods.json)")
    if not os.path.isfile(script):
        problems.append(f"code not found: {script} (git submodule update --init?)")
    for key, value in cfg["paths"].items():
        if (key.endswith("_ckpt") or key.endswith("_dir")) and any("{" + key + "}" in a for a in method["args"]):
            if not os.path.exists(resolve_path(value)):
                problems.append(f"not found: {resolve_path(value)} ('{key}' in methods.json)")
    return problems


def write_inputs(job):
    for path, data in ((job["job_file"], job["record"]), (job["prompt_file"], job["prompt_relay"])):
        if path == job["prompt_file"] and not job["uses_prompt_file"]:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- running on all GPUs
def gpu_groups(gpus, per_job):
    if per_job < 1 or len(gpus) < per_job:
        raise ConfigError(f"--gpus_per_job {per_job} does not fit --gpus {','.join(gpus)}")
    return [gpus[i:i + per_job] for i in range(0, len(gpus) - per_job + 1, per_job)]


def start_task(task, cfg, group, master_port):
    cmd, cwd, visible = task_command(task, cfg, group, master_port)
    for job in task["jobs"]:
        write_inputs(job)
    os.makedirs(os.path.dirname(task["log"]), exist_ok=True)
    log = open(task["log"], "w", encoding="utf-8")
    log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + f"\n(cwd {cwd}, CUDA_VISIBLE_DEVICES={visible})\n\n")
    log.flush()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log, visible


def describe_task(task):
    ids = [j["id"] for j in task["jobs"]]
    return f"{task['method']} / " + (ids[0] if len(ids) == 1 else f"{len(ids)} videos ({ids[0]} ... {ids[-1]})")


def run_all_tasks(tasks, cfg, groups, poll_s=5.0):
    """Run tasks on the GPU groups, each group taking the next waiting task when it is free."""
    waiting = list(tasks)
    running = {}                 # group index -> (task, proc, log, start, visible)
    made, failed = [], []
    total = len(tasks)
    finished = 0

    def stop_all(*_):
        for _t, proc, _l, _s, _v in running.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGTERM, stop_all)
    try:
        while waiting or running:
            for gi, group in enumerate(groups):
                if gi in running or not waiting:
                    continue
                task = waiting.pop(0)
                port = 29500 + (int(group[0]) if group[0].isdigit() else gi)
                proc, log, visible = start_task(task, cfg, group, port)
                running[gi] = (task, proc, log, time.time(), visible)
                print(f"[generate] start  {describe_task(task)} on GPU {visible}  (log: {task['log']})", flush=True)
            for gi in list(running):
                task, proc, log, start, visible = running[gi]
                if proc.poll() is None:
                    continue
                log.close()
                del running[gi]
                finished += 1
                ok = [j for j in task["jobs"] if os.path.isfile(j["output"])]
                bad = [j for j in task["jobs"] if not os.path.isfile(j["output"])]
                made += ok
                failed += bad
                minutes = (time.time() - start) / 60
                status = "done" if not bad and proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"[generate] {status}  {describe_task(task)} on GPU {visible} in {minutes:.1f} min"
                      + (f"; missing: {', '.join(j['id'] for j in bad)}" if bad else "")
                      + f"  [{finished}/{total} tasks]", flush=True)
            if running:
                time.sleep(poll_s)
    except KeyboardInterrupt:
        for _t, proc, _l, _s, _v in running.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        print("\n[generate] stopped; running jobs were terminated. Run the same command again to continue "
              "(finished videos are kept).")
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGTERM, old)
    return made, failed


def show_status(jobs, names):
    print(f"[generate] status ({time.strftime('%H:%M:%S')})")
    for name in names:
        mj = [j for j in jobs if j["method"] == name]
        if not mj:
            continue
        done = [j for j in mj if os.path.isfile(j["output"])]
        started = [j for j in mj if not os.path.isfile(j["output"]) and os.path.isfile(j["log"])]
        print(f"  {name:32s} {len(done):4d} / {len(mj):<4d} made"
              + (f", {len(started)} started or failed" if started else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="the prompt-suite JSON (same file run_all.py scores with)")
    p.add_argument("--config", default=os.path.join(HERE, "methods.json"), help="methods and paths")
    p.add_argument("--methods", default=None, help="comma-separated methods to run (default: every enabled one)")
    p.add_argument("--ids", default=None, help="comma-separated entry ids to run (default: all)")
    p.add_argument("--out_root", default="outputs", help="videos go to OUT_ROOT/<method>/<file_name>")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0",
                   help="GPU ids to use, e.g. 0,1,2,3 (default: $CUDA_VISIBLE_DEVICES, else 0)")
    p.add_argument("--gpus_per_job", type=int, default=1,
                   help="GPUs per video (default 1: one video per GPU, all GPUs busy at once). Methods with "
                        "'parallel': 'wan' are sharded over 2, 4 or 8 (one model split over the GPUs, faster); "
                        "other methods use the first GPU of the group")
    p.add_argument("--batch_size", type=int, default=8,
                   help="videos a batch method makes per model load (default 8)")
    p.add_argument("--interleave", action="store_true",
                   help="alternate the methods' batches (A 1-8, B 1-8, A 9-16, ...) so a run stopped early still "
                        "has matching videos from every method; default: all of A, then all of B")
    p.add_argument("--print_only", action="store_true", help="print the commands; run and write nothing")
    p.add_argument("--status", action="store_true", help="show how many videos each method has made; run nothing")
    p.add_argument("--force", action="store_true", help="regenerate videos that already exist")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")
    out_root = os.path.abspath(args.out_root)
    entries = S.load_suite(args.prompts, videos_dir=out_root)
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        unknown = [i for i in wanted if i not in {e["id"] for e in entries}]
        if unknown:
            raise SystemExit(f"unknown ids {unknown} in {args.prompts}")
        entries = [e for e in entries if e["id"] in wanted]

    names = list(cfg["methods"])
    if args.methods:
        asked = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in asked if m not in cfg["methods"]]
        if unknown:
            raise SystemExit(f"unknown methods {unknown}; methods.json has {names}")
        names = asked
    disabled = [m for m in names if not cfg["methods"][m].get("enabled", True)]
    names = [m for m in names if m not in disabled]
    if disabled:
        print(f"[generate] disabled in methods.json, skipped: {', '.join(disabled)}")
    try:
        jobs, skipped = plan(entries, cfg, names, out_root)
        groups = gpu_groups(gpus, args.gpus_per_job)
    except ConfigError as err:
        raise SystemExit(f"[generate] {err}")
    if args.status:
        show_status(jobs, names)
        return 0

    n_image = sum(bool([i for i in e["images"] if i]) for e in entries)
    print(f"[generate] {len(entries)} prompt(s) in {os.path.basename(args.prompts)}: "
          f"{len(entries) - n_image} text-only, {n_image} with images")
    idle = []
    for name in names:
        n_jobs = sum(j["method"] == name for j in jobs)
        n_skip = sum(s[0] == name for s in skipped)
        if n_jobs == 0:
            idle.append(name)
            continue
        print(f"[generate] {name}: {n_jobs} video(s)" + (f" ({n_skip} prompt(s) not for this method)"
                                                         if n_skip else ""))
    if idle:
        print(f"[generate] nothing to make for: {', '.join(idle)} (no prompt has the images they need)")

    pending = [j for j in jobs if args.force or not os.path.isfile(j["output"])]
    kept = len(jobs) - len(pending)
    tasks = make_tasks(pending, cfg, args.batch_size, interleave=args.interleave)
    try:
        if args.print_only:
            for task in tasks:
                group = groups[0]
                cmd, cwd, visible = task_command(task, cfg, group, 29500)
                print(f"\n# {describe_task(task)}  (cwd {cwd}, CUDA_VISIBLE_DEVICES={visible})")
                print(" ".join(shlex.quote(c) for c in cmd))
            print(f"\n[generate] {len(tasks)} task(s) for {len(pending)} video(s) ({kept} already made) on "
                  f"{len(groups)} GPU group(s): {' '.join(','.join(g) for g in groups)}")
            return 0
        blocked = {}
        for name in {t["method"] for t in tasks}:
            problems = missing_files(name, cfg["methods"][name], cfg)
            if problems:
                blocked[name] = problems
    except ConfigError as err:
        raise SystemExit(f"[generate] {err}")
    for m, probs in blocked.items():
        print(f"[generate] {m} cannot run:\n  " + "\n  ".join(probs))
    if blocked:
        return 1
    if args.force:
        for job in pending:
            for path in (job["output"], os.path.splitext(job["output"])[0] + ".meta.json"):
                if os.path.isfile(path):
                    os.remove(path)
    print(f"[generate] {len(pending)} video(s) to make in {len(tasks)} task(s), {kept} already made; "
          f"{len(groups)} GPU group(s): {' '.join(','.join(g) for g in groups)}", flush=True)
    made, failed = run_all_tasks(tasks, cfg, groups)
    print(f"[generate] {len(made)} generated, {kept} kept, {len(failed)} failed; videos in {out_root}/<method>/")
    for job in failed:
        print(f"  failed: {job['method']} / {job['id']} (see {os.path.dirname(job['log'])})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
