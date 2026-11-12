#!/usr/bin/env python
"""Generate every method's video for every prompt in the suite JSON, ready for run_all.py.

    python vbench_eval/generate_suite.py --prompts PROMPTS.json [--methods a,b] [--ids p001,v001] [--gpus 0,1]
                                         [--out_root outputs] [--config vbench_eval/methods.json]
                                         [--print_only] [--force]

Each method in methods.json says which code, Python environment, checkpoint and arguments it uses, and which
entries it applies to (text-only entries, or entries with images). Videos go to OUT_ROOT/<method>/<file_name>,
the same file name for every method, with a <name>.meta.json beside each (generation time, and for image
entries the frame where Img2 starts) and the generation log in OUT_ROOT/<method>/logs/. Videos that already
exist are kept (--force regenerates). Methods with "enabled": false are listed and skipped.
Run from anywhere; any Python 3 works (this script only starts the methods' own environments).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from vbeval import suite as S  # noqa: E402

RUNS_ON = ("text", "first_image", "two_images", "all")
PARALLEL = ("wan", "none")
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
        for key in ("runs_on", "python", "code_dir", "script", "parallel", "args"):
            if key not in m:
                raise ConfigError(f"method {name}: missing '{key}'")
        if m["runs_on"] not in RUNS_ON:
            raise ConfigError(f"method {name}: runs_on must be one of {RUNS_ON}")
        if m["parallel"] not in PARALLEL:
            raise ConfigError(f"method {name}: parallel must be one of {PARALLEL}")
        if m.get("record_switch") not in (None, "image_latents"):
            raise ConfigError(f"method {name}: record_switch must be 'image_latents' or absent")
        if not isinstance(m["args"], list) or not all(isinstance(a, str) for a in m["args"]):
            raise ConfigError(f"method {name}: args must be a list of strings")
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


# ---------------------------------------------------------------- what runs on what
def joined_prompt(entry):
    return " ".join(p.strip() for p in [entry["global_prompt"], *entry["local_prompts"]] if p and p.strip())


def applies(method, entry):
    """None if the method runs on this entry, else the reason it does not."""
    images = [i for i in entry["images"] if i]
    kind = method["runs_on"]
    if kind == "text" and images:
        return "entry has images (text-only method)"
    if kind == "first_image" and not images:
        return "entry has no image"
    if kind == "two_images" and (len(images) != 2 or len(entry["local_prompts"]) != 2):
        return "needs exactly 2 images and 2 local prompts"
    return None


def entry_values(entry, output, prompt_file):
    values = {"output": output, "prompt_file": prompt_file, "prompt_joined": joined_prompt(entry),
              "global_prompt": entry["global_prompt"]}
    for k, text in enumerate(entry["local_prompts"], start=1):
        values[f"local_{k}"] = text
    for k, image in enumerate([i for i in entry["images"] if i], start=1):
        values[f"image_{k}"] = os.path.abspath(image)
    for k, latent in enumerate(entry["image_latents"] or [], start=1):
        values[f"image_latent_{k}"] = str(latent)
    return values


def switch_latents(method, entry, frame_num):
    if method.get("record_switch") != "image_latents" or not entry["image_latents"]:
        return []
    latents = entry["image_latents"][1:]
    n_latents = (int(frame_num) - 1) // 4 + 1
    for latent in latents:
        if not 1 <= latent < n_latents:
            raise ConfigError(f"{entry['id']}: image latent {latent} is outside 1..{n_latents - 1} "
                              f"for {frame_num} frames")
    return latents


def build_command(name, method, cfg, entry, out_root, gpus, master_port):
    """The full command (timed_run wrapper included) and its working directory for one video."""
    out_dir = os.path.join(out_root, name)
    output = os.path.join(out_dir, entry["file_name"])
    prompt_file = os.path.join(out_root, "_prompt_files", f"{entry['id']}.json")
    values = {**{k: path_value(k, v) for k, v in cfg["paths"].items()},
              **{k: str(v) for k, v in cfg["settings"].items()},
              **entry_values(entry, output, prompt_file)}
    if (method.get("record_switch") == "image_latents" and len([i for i in entry["images"] if i]) > 1
            and not entry["image_latents"]):
        raise ConfigError(f"{entry['id']}: {name} needs 'image_latents' (where each image is placed), "
                          "e.g. [0, 14]")
    where = f"method {name}"
    python = resolve_path(fill(method["python"], values, where))
    code_dir = resolve_path(method["code_dir"])
    args = [fill(a, values, where) for a in method["args"]]
    if method["parallel"] == "wan":
        n = len(gpus)
        if n not in WAN_GPU_COUNTS:
            raise ConfigError(f"Wan runs on {WAN_GPU_COUNTS} GPUs, got {n} ({','.join(gpus)})")
        if n == 1:
            cmd = [python, method["script"], *args, "--offload_model", "True", "--convert_model_dtype"]
        else:
            cmd = [python, "-m", "torch.distributed.run", f"--nproc_per_node={n}", f"--master_port={master_port}",
                   method["script"], *args, "--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(n)]
        visible = gpus
    else:
        cmd = [python, method["script"], *args]
        visible = gpus[:1]
    frame_num = cfg["settings"].get("wan_frame_num", 81)
    latents = switch_latents(method, entry, frame_num)
    wrapper = [sys.executable, os.path.join(HERE, "timed_run.py"), "--video", output]
    if latents:
        wrapper += ["--switch_latents", *[str(l) for l in latents]]
    return {"method": name, "id": entry["id"], "output": output, "prompt_file": prompt_file,
            "uses_prompt_file": any("{prompt_file}" in a for a in method["args"]),
            "cwd": code_dir, "python": python, "script": os.path.join(code_dir, method["script"]),
            "gpus": ",".join(visible), "command": wrapper + ["--"] + cmd,
            "log": os.path.join(out_dir, "logs", f"{entry['id']}.log")}


def missing_files(job, cfg, method):
    """Things that must exist before a method can run (checked once per method)."""
    problems = []
    if not os.path.isfile(job["python"]):
        problems.append(f"Python not found: {job['python']} (set 'paths' in methods.json)")
    if not os.path.isfile(job["script"]):
        problems.append(f"code not found: {job['script']} (git submodule update --init?)")
    for key, value in cfg["paths"].items():
        if key.endswith("_ckpt") and any("{" + key + "}" in a for a in method["args"]):
            if not os.path.isdir(resolve_path(value)):
                problems.append(f"checkpoint folder not found: {resolve_path(value)} ('{key}' in methods.json)")
    return problems


def plan(entries, cfg, methods, out_root, gpus, master_port):
    jobs, skipped = [], []
    for name in methods:
        method = cfg["methods"][name]
        for entry in entries:
            reason = applies(method, entry)
            if reason:
                skipped.append((name, entry["id"], reason))
                continue
            jobs.append(build_command(name, method, cfg, entry, out_root, gpus, master_port))
    return jobs, skipped


# ---------------------------------------------------------------- running
def write_prompt_file(job, entry):
    os.makedirs(os.path.dirname(job["prompt_file"]), exist_ok=True)
    with open(job["prompt_file"], "w", encoding="utf-8") as f:
        json.dump({"global_prompt": entry["global_prompt"], "local_prompts": entry["local_prompts"]},
                  f, indent=2, ensure_ascii=False)


def run_job(job, entry):
    os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
    if job["uses_prompt_file"]:
        write_prompt_file(job, entry)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=job["gpus"])
    start = time.time()
    with open(job["log"], "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(c) for c in job["command"]) + f"\n(cwd {job['cwd']}, "
                  f"CUDA_VISIBLE_DEVICES={job['gpus']})\n\n")
        log.flush()
        code = subprocess.run(job["command"], cwd=job["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return code, time.time() - start


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="the prompt-suite JSON (same file run_all.py scores with)")
    p.add_argument("--config", default=os.path.join(HERE, "methods.json"), help="methods and paths")
    p.add_argument("--methods", default=None, help="comma-separated methods to run (default: every enabled one)")
    p.add_argument("--ids", default=None, help="comma-separated entry ids to run (default: all)")
    p.add_argument("--out_root", default="outputs", help="videos go to OUT_ROOT/<method>/<file_name>")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1",
                   help="GPU ids to use (default: $CUDA_VISIBLE_DEVICES, else 0,1)")
    p.add_argument("--master_port", type=int, default=None, help="torchrun port (default 29500 + first GPU id)")
    p.add_argument("--print_only", action="store_true", help="print the commands; run and write nothing")
    p.add_argument("--force", action="store_true", help="regenerate videos that already exist")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")
    master_port = args.master_port or 29500 + (int(gpus[0]) if gpus[0].isdigit() else 0)
    out_root = os.path.abspath(args.out_root)
    entries = S.load_suite(args.prompts, videos_dir=out_root)
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        unknown = [i for i in wanted if i not in {e["id"] for e in entries}]
        if unknown:
            raise SystemExit(f"unknown ids {unknown} in {args.prompts}")
        entries = [e for e in entries if e["id"] in wanted]
    by_id = {e["id"]: e for e in entries}

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
        jobs, skipped = plan(entries, cfg, names, out_root, gpus, master_port)
    except ConfigError as err:
        raise SystemExit(f"[generate] {err}")
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
        print(f"[generate] nothing to make for: {', '.join(idle)} (they take the other kind of prompt)")

    if args.print_only:
        for job in jobs:
            print(f"\n# {job['method']} / {job['id']}  (cwd {job['cwd']}, CUDA_VISIBLE_DEVICES={job['gpus']})")
            print(" ".join(shlex.quote(c) for c in job["command"]))
        return 0

    problems = {}
    for job in jobs:
        if job["method"] not in problems:
            problems[job["method"]] = missing_files(job, cfg, cfg["methods"][job["method"]])
    blocked = {m: p for m, p in problems.items() if p}
    for m, p in blocked.items():
        print(f"[generate] {m} cannot run:\n  " + "\n  ".join(p))
    if blocked:
        return 1

    done, kept, failed = 0, 0, []
    total = len(jobs)
    for i, job in enumerate(jobs, start=1):
        tag = f"[generate] {i}/{total} {job['method']} / {job['id']}"
        if os.path.isfile(job["output"]) and not args.force:
            print(f"{tag}: exists, kept (--force to regenerate)")
            kept += 1
            continue
        if args.force and os.path.isfile(job["output"]):
            os.remove(job["output"])
        print(f"{tag}: running on GPU {job['gpus']} (log: {job['log']})", flush=True)
        code, seconds = run_job(job, by_id[job["id"]])
        if code == 0 and os.path.isfile(job["output"]):
            print(f"{tag}: done in {seconds / 60:.1f} min")
            done += 1
        else:
            print(f"{tag}: FAILED (exit {code}); see {job['log']}")
            failed.append(job)
    print(f"[generate] {done} generated, {kept} kept, {len(failed)} failed; videos in {out_root}/<method>/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
