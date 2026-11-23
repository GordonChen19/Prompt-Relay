#!/usr/bin/env python
"""MiniMax H3 (src/MiniMax-H3, Prompt Relay version): load the pipeline once, make many videos on one GPU.

    cd src/MiniMax-H3 && python vbench_eval/drivers/h3_batch.py --model_path H3_WEIGHTS
                         [--prompt_relay | --edit switchcraft --switchcraft_dir DIR | --edit ts_attn]
                         [--frame_num 192] [--height 480] [--width 832] [--steps 50] [--cpu_offload True]
                         --jobs JOB.json [JOB.json ...]

Does what generate_prompt_relay.py does (same frame alignment, seed generator, Prompt Relay hook, saving
with audio at 24 fps), without reloading the model for every video:
  plain                 the job's prompt_joined as one prompt
  --prompt_relay        the job's Prompt Relay JSON (its frame_intervals are already on H3's timeline)
  --edit switchcraft    plain H3's prompt, with SwitchCraft (CVPR 2026) in H3's attention (h3_edits.py)
  --edit ts_attn        plain H3's prompt, with TS-Attn (ICLR 2026) in H3's attention (h3_edits.py)
The two edits get the same inputs as their Wan 2.2 runs: events = the entry's event_words, else its local
prompts; each event's share from the event frames; TS-Attn's subject = the entry's subject, else the same
guess ts_attn.py makes. Blocks and steps outside the method's schedule run plain H3 exactly.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobio  # noqa: E402
import h3_edit_plan as EP  # noqa: E402

EDITS = ("none",) + EP.METHODS


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--prompt_relay", action="store_true")
    p.add_argument("--edit", default="none", choices=EDITS, help="TS-Attn or SwitchCraft inside H3's attention")
    p.add_argument("--switchcraft_dir", default=None, help="the authors' SwitchCraft repo (their edit code)")
    p.add_argument("--edit_steps_frac", type=float, default=None,
                   help="share of the steps with the edit on (default: SwitchCraft 0.4, TS-Attn 0.2)")
    p.add_argument("--sc_edit_blocks", type=int, default=None,
                   help="SwitchCraft: edit blocks 0..N-1 (default: half of H3's blocks, as their 20 of 40)")
    p.add_argument("--ts_logits", default="attn", choices=("attn", "raw"),
                   help="TS-Attn statistics from H3's attention logits (attn) or from q/k before norm and "
                        "rotary (raw, their literal to_q/to_k)")
    p.add_argument("--ts_stats", default="released", choices=("released", "own"),
                   help="TS-Attn: event k's max/min/mean from the rows their released code reads (starting at row "
                        "k, as the Wan 2.2 TS-Attn baseline ran) or from each row's own logits")
    p.add_argument("--frame_num", type=int, default=124)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--query_chunk_size", type=int, default=128)
    p.add_argument("--cpu_offload", default="True")
    p.add_argument("--memory_reserve_margin", default=None,
                   help="GPU memory kept free during CPU offload, e.g. 12GB (default: diffusers' own)")
    p.add_argument("--jobs", nargs="+", required=True)
    return p


def check_args(p, args):
    if args.height % 32 or args.width % 32:
        p.error("height and width must be multiples of 32")
    if args.edit != "none" and args.prompt_relay:
        p.error("--edit and --prompt_relay cannot be combined")
    if args.edit == EP.SWITCHCRAFT and not args.switchcraft_dir:
        p.error("--edit switchcraft needs --switchcraft_dir")
    if args.query_chunk_size < 1:
        p.error("--query_chunk_size must be positive")


def edit_inputs(job):
    """The events, their shares and (TS-Attn) the subject, as the Wan 2.2 drivers take them."""
    from ts_attn import guess_subject
    prompt = job["prompt_joined"]
    events = job.get("event_words") or job["local_prompts"]
    subject = job.get("subject") or [guess_subject(job["global_prompt"] or job["local_prompts"][0])]
    if isinstance(subject, str):
        subject = [subject]
    return prompt, list(events), EP.normalize_ratios(jobio.event_durations(job)), list(subject)


def token_offsets(tokenizer, prompt):
    """Character span of every token H3 encodes for this prompt (t2va: the prompt verbatim, no special tokens)."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) != len(offsets):
        raise ValueError("tokenizer offsets do not match its tokens")
    return [tuple(o) for o in offsets]


def denoise_forwards(pipe, steps):
    """How many transformer forwards H3 makes for `steps` (one per scheduler timestep: 50 steps give 49).
    The edit windows count these forwards, as the authors' rules count their sampler's timesteps."""
    try:
        pipe.scheduler.set_timesteps(steps)          # the pipeline sets it again for every video
        return len(pipe.scheduler.timesteps)
    except Exception as err:  # noqa: BLE001
        print("[h3_batch] could not read H3's timesteps (%s); assuming %d forwards" % (err, steps - 1), flush=True)
        return steps - 1


def edit_schedule(edit, pipe, steps, n_blocks, frac=None, sc_blocks=None):
    """(forwards, edit steps, edit blocks) for this H3 pipeline."""
    forwards = denoise_forwards(pipe, steps)
    return forwards, EP.active_steps(edit, forwards, frac), EP.active_blocks(edit, n_blocks, sc_blocks)


def edit_check(calls, active, forwards, until, n_blocks):
    """None if the edit ran where it should, else a message (recorded next to the video, never fatal)."""
    want = until * n_blocks
    if calls == forwards and active == want:
        return None
    return "edit ran in %d block-steps over %d forwards; expected %d over %d" % (active, calls, want, forwards)


def main(argv=None):
    p = get_parser()
    args = p.parse_args(argv)
    check_args(p, args)

    jobio.code_dir_on_path()
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames, video_latent_num_frames
    from diffusers.utils.export_utils import encode_video
    from h3_prompt_relay import PromptRelayConfig, enable_prompt_relay

    frames = align_num_frames(args.frame_num, 17, 5)
    if not 5 <= frames / 24 <= 15:
        raise SystemExit("H3 output must last 5-15 seconds; %d frames aligned to %d" % (args.frame_num, frames))
    internal = video_latent_num_frames(frames, 17, 5)
    jobs = jobio.load_jobs(args.jobs)

    t0 = time.time()
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(args.model_path, workflow="t2va", components_manager=manager)
    enable_prompt_relay(pipe)
    pipe.load_components(dtype=torch.bfloat16)
    if str2bool(args.cpu_offload):
        if args.memory_reserve_margin:
            manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=args.memory_reserve_margin)
        else:
            manager.enable_auto_cpu_offload(device="cuda")
    else:
        pipe.to("cuda")

    edit_state, edit_info = None, {}
    if args.edit != "none":
        import h3_edits as E
        transformer = getattr(pipe, "transformer", None) or manager.get_one(name="transformer")
        n_blocks = len(transformer.transformer_blocks)
        edit_fn, sc_sources = None, None
        if args.edit == EP.SWITCHCRAFT:
            edit_fn, sc_sources = E.load_switchcraft_edit(args.switchcraft_dir)
        forwards, until, blocks = edit_schedule(args.edit, pipe, args.steps, n_blocks, args.edit_steps_frac,
                                                args.sc_edit_blocks)
        edit_state = E.EditState(args.edit, until, blocks, chunk_size=args.query_chunk_size, edit_fn=edit_fn,
                                 ts_logits=args.ts_logits, ts_stats=args.ts_stats)
        E.install(transformer, edit_state)
        edit_info = {"edit": args.edit, "edit_steps": until, "of_steps": forwards, "edit_blocks": blocks,
                     "of_blocks": n_blocks}
        if args.edit == EP.TS_ATTN:
            edit_info.update(ts_logits=args.ts_logits, ts_stats=args.ts_stats)
        if sc_sources:
            edit_info["switchcraft_file"] = E.switchcraft_model_py(args.switchcraft_dir)
        tokenizer = pipe.tokenizer
    load_s = time.time() - t0
    print("[h3_batch] loaded in %.0f s; %d frames (%d internal); %d job(s)%s" % (
        load_s, frames, internal, len(jobs),
        "" if not edit_info else "; %s on steps 0-%d of %d, blocks %s" % (
            args.edit, edit_info["edit_steps"] - 1, edit_info["of_steps"], edit_info["edit_blocks"])), flush=True)

    def make_one(job):
        inputs = {"num_frames": frames, "height": args.height, "width": args.width,
                  "num_inference_steps": args.steps, "generator": torch.Generator().manual_seed(int(job["seed"])),
                  "query_chunk_size": args.query_chunk_size}
        record = {}
        if args.prompt_relay:
            relay = PromptRelayConfig.from_mapping(job["prompt_relay"])
            relay.frame_intervals(internal)            # same check as generate_prompt_relay.py
            inputs["prompt_relay_config"] = relay.to_mapping()
        else:
            inputs["prompt"] = job["prompt_joined"]
        if edit_state is not None:
            prompt, events, shares, subject = edit_inputs(job)
            offsets = token_offsets(tokenizer, prompt)
            plan = EP.video_plan(args.edit, prompt, offsets, events, shares,
                                 subject if args.edit == EP.TS_ATTN else None)
            edit_state.begin_video(plan)
            record = {"events": events, "event_shares": [round(x, 4) for x in shares],
                      "event_tokens": [len(c) for c in plan["event_cols"]]}
            if args.edit == EP.TS_ATTN:
                record.update(subject=subject, subject_tokens=[len(c) for c in plan["subject_cols"]])
            else:
                record["frame_split"] = EP.sc_frame_sizes(internal, shares)
        try:
            results = pipe(**inputs, output=["videos", "audio", "sampling_rate"])
        finally:
            if edit_state is not None:
                calls, active = edit_state.calls, edit_state.active_calls
                edit_state.end_video()
        encode_video(results["videos"][0], fps=24, output_path=str(job["output"]),
                     audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
        if edit_state is not None:                     # the video is saved first: a check never discards it
            problem = edit_check(calls, active, edit_info["of_steps"], edit_info["edit_steps"],
                                 len(edit_info["edit_blocks"]))
            record.update(edited_block_steps=active, forwards=calls)
            if problem:
                record["edit_check"] = problem
                print("[h3_batch] %s: WARNING %s" % (job["id"], problem), flush=True)
        settings = {"prompt_relay": args.prompt_relay, "frames": frames, "internal_frames": internal,
                    "height": args.height, "width": args.width, "steps": args.steps, "seed": int(job["seed"])}
        settings.update(edit_info)
        out = {"settings": settings}
        if record:
            out["inputs"] = record
        return out

    return jobio.run_batch(jobs, make_one, "h3_batch", load_seconds=load_s)


if __name__ == "__main__":
    sys.exit(main())
