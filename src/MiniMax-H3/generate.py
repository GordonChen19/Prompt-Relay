"""MiniMax H3 generation with optional overlapping Prompt Relay/sliding windows."""

import argparse
import json
from pathlib import Path


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false or 1/0.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="MiniMaxAI/MiniMax-H3")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt_filepath", type=Path)
    parser.add_argument("--image", help="Optional first keyframe")
    parser.add_argument("--last_image", help="Optional last keyframe")
    parser.add_argument("--frame_num", type=int, default=124)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sliding_window", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--window_length", type=int, default=31)
    parser.add_argument("--window_stride", type=int, default=16)
    parser.add_argument("--query_chunk_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_offload", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--output_path", type=Path, default=Path("outputs/minimax_h3.mp4"))
    args = parser.parse_args(argv)
    if args.frame_num < 1 or args.num_inference_steps < 2 or args.query_chunk_size < 1:
        parser.error("frame_num and query_chunk_size must be positive; num_inference_steps must be >= 2.")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        parser.error("height and width must be positive multiples of 32.")
    if args.sliding_window and not 0 < args.window_stride <= args.window_length:
        parser.error("Require 0 < window_stride <= window_length.")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames, video_latent_num_frames
    from diffusers.modular_pipelines.minimax_h3.before_denoise import _temporal_position_grid
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video
    from h3_prompt_relay import enable_prompt_relay
    from h3_prompt_relay.schedule import prepare_schedule

    frames = align_num_frames(args.frame_num, 17, 5)
    if not 5 <= frames / 24 <= 15:
        raise ValueError(f"H3 aligned output must last 5–15 seconds; requested {args.frame_num}, aligned {frames} frames.")
    config = None
    if args.prompt_filepath:
        config = json.loads(args.prompt_filepath.read_text(encoding="utf-8-sig"))
        prepare_schedule(config, _temporal_position_grid(video_latent_num_frames(frames, 17, 5), 0) / 40, frames / 24)
    workflow = "fl2va" if args.image or args.last_image else "t2va"
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(args.model_path, workflow=workflow, components_manager=manager)
    enable_prompt_relay(pipe)
    pipe.load_components(dtype=torch.bfloat16)
    if args.cpu_offload:
        manager.enable_auto_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)
    inputs = dict(num_frames=frames, height=args.height, width=args.width,
                  num_inference_steps=args.num_inference_steps, generator=torch.Generator().manual_seed(args.seed),
                  sliding_window=args.sliding_window, window_length=args.window_length,
                  window_stride=args.window_stride, query_chunk_size=args.query_chunk_size)
    if config is not None:
        inputs["prompt_relay_config"] = config
    else:
        inputs["prompt"] = args.prompt
    if args.image:
        inputs["image"] = load_image(args.image)
    if args.last_image:
        inputs["last_image"] = load_image(args.last_image)
    results = pipe(**inputs, output=["videos", "audio", "sampling_rate"])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(results["videos"][0], fps=24, output_path=str(args.output_path),
                 audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
    metadata = dict(vars(args), actual_num_frames=frames, fps=24, prompt_relay_config=config)
    args.output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
