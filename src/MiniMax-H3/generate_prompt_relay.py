"""Generate MiniMax H3 video with optional Wan-equivalent Prompt Relay."""

import argparse
import json
import os
from pathlib import Path


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false or 1/0")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=".")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt_filepath", type=Path)
    parser.add_argument("--image", help="Optional first keyframe")
    parser.add_argument("--last_image", help="Optional last keyframe")
    parser.add_argument("--frame_num", type=int, default=124)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional explicit canvas height; provide together with --width",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional explicit canvas width; provide together with --height",
    )
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_chunk_size", type=int, default=128)
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Number of torchrun ranks used for packed-sequence parallelism",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--memory_reserve_margin",
        default="12GB",
        help="GPU memory kept free by ComponentsManager during CPU offload",
    )
    parser.add_argument(
        "--cpu_offload",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--output_path", type=Path, default=Path("outputs/h3_prompt_relay.mp4")
    )
    args = parser.parse_args(argv)
    if args.frame_num < 1 or args.num_inference_steps < 2:
        parser.error("frame_num must be positive and num_inference_steps must be >= 2")
    if args.query_chunk_size < 1:
        parser.error("query_chunk_size must be positive")
    if args.ulysses_size < 1:
        parser.error("ulysses_size must be positive")
    if (args.height is None) != (args.width is None):
        parser.error("height and width must be provided together")
    if args.height is not None and (
        args.height <= 0
        or args.width <= 0
        or args.height % 32
        or args.width % 32
    ):
        parser.error("height and width must be positive multiples of 32")
    return args


def main(argv=None):
    args = parse_args(argv)

    import torch
    import torch.distributed as dist
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
    )
    from diffusers.models import ContextParallelConfig
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        align_num_frames,
        video_latent_num_frames,
    )
    from diffusers.modular_pipelines.minimax_h3.decoders import (
        MiniMaxH3AudioDecodeStep,
        MiniMaxH3VideoDecodeStep,
    )
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    from h3_prompt_relay import PromptRelayConfig, enable_prompt_relay

    distributed = args.ulysses_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU Prompt Relay requires CUDA")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != args.ulysses_size:
            raise ValueError(
                f"--ulysses_size={args.ulysses_size} requires torchrun with "
                f"WORLD_SIZE={args.ulysses_size}; got {world_size}"
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
    frames = align_num_frames(args.frame_num, 17, 5)
    if not 5 <= frames / 24 <= 15:
        raise ValueError(
            "H3 aligned output must last 5–15 seconds; "
            f"requested {args.frame_num}, aligned {frames} frames"
        )

    internal_frames = video_latent_num_frames(frames, 17, 5)
    relay_config = None
    if args.prompt_filepath:
        relay_config = PromptRelayConfig.from_json(args.prompt_filepath)
        relay_config.frame_intervals(internal_frames)

    workflow = "fl2va" if args.image or args.last_image else "t2va"
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        args.model_path,
        workflow=workflow,
        components_manager=manager,
    )
    enable_prompt_relay(pipe)
    if distributed and dist.get_rank() != 0:
        decoder_types = (MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep)

        def remove_decoders(parent):
            for name, block in list(parent.sub_blocks.items()):
                if isinstance(block, decoder_types):
                    del parent.sub_blocks[name]
                elif block.sub_blocks:
                    remove_decoders(block)

        remove_decoders(pipe._blocks)
    pipe.load_components(dtype=torch.bfloat16)
    if distributed:
        transformer = manager.get_one(name="transformer")
        transformer.set_attention_backend("native")
        cp_plan = {
            "rope": {
                0: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
            },
            "transformer_blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3
                ),
            },
            "transformer_blocks.*": {
                "adaln_indices": ContextParallelInput(
                    split_dim=0, expected_dims=1
                ),
            },
            "norm_out": {
                "timestep_indices": ContextParallelInput(
                    split_dim=0, expected_dims=1
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
            "audio_proj_out": ContextParallelOutput(
                gather_dim=1, expected_dims=3
            ),
        }
        transformer.enable_parallelism(
            config=ContextParallelConfig(
                ulysses_degree=args.ulysses_size,
                ulysses_anything=True,
            ),
            cp_plan=cp_plan,
        )
    if args.cpu_offload:
        manager.enable_auto_cpu_offload(
            device=args.device,
            memory_reserve_margin=args.memory_reserve_margin,
        )
    else:
        pipe.to(args.device)

    inputs = {
        "num_frames": frames,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator().manual_seed(args.seed),
        "query_chunk_size": args.query_chunk_size,
    }
    if args.height is not None:
        inputs["height"] = args.height
        inputs["width"] = args.width
    if relay_config is not None:
        inputs["prompt_relay_config"] = relay_config.to_mapping()
    else:
        inputs["prompt"] = args.prompt
    if args.image:
        inputs["image"] = load_image(args.image)
    if args.last_image:
        inputs["last_image"] = load_image(args.last_image)

    rank_zero = not distributed or dist.get_rank() == 0
    requested_outputs = (
        ["videos", "audio", "sampling_rate"]
        if rank_zero
        else ["latents", "audio_latents"]
    )
    results = pipe(**inputs, output=requested_outputs)
    if rank_zero:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            results["videos"][0],
            fps=24,
            output_path=str(args.output_path),
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )
        metadata = {
            **vars(args),
            "actual_num_frames": frames,
            "fps": 24,
            "prompt_relay_config": (
                relay_config.to_mapping() if relay_config is not None else None
            ),
        }
        args.output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
