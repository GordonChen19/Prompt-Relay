"""Optional blocks around the native H3 encoders and complete denoising loop."""

import torch

from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep, MiniMaxH3Ref2VADenoiseStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

from .attention import SlidingWindowConfig, prepare_attention_state, use_relay_attention
from .schedule import build_prompt_text


class RelayPromptStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return "Build the complete Prompt Relay text before the native H3 encoder."

    @property
    def inputs(self):
        return [InputParam("prompt", type_hint=str, default=None),
                InputParam("prompt_relay_config", type_hint=dict, default=None),
                InputParam("sliding_window", type_hint=bool, default=False),
                InputParam("window_length", type_hint=int, default=31),
                InputParam("window_stride", type_hint=int, default=16),
                InputParam("query_chunk_size", type_hint=int, default=128)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("prompt", type_hint=str)]

    def __call__(self, components, state):
        if type(state.get("sliding_window")) is not bool:
            raise ValueError("sliding_window must be a boolean.")
        if state.get("sliding_window"):
            SlidingWindowConfig(state.get("window_length"), state.get("window_stride"))
        chunk = state.get("query_chunk_size")
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("query_chunk_size must be a positive integer.")
        config, prompt = state.get("prompt_relay_config"), state.get("prompt")
        if config is not None:
            if prompt not in (None, ""):
                raise ValueError("Provide prompt_relay_config or prompt, not both.")
            prompt, _ = build_prompt_text(config)
            state.set("prompt", prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty prompt or prompt_relay_config.")
        return components, state


class _RelayLoopMixin:
    @torch.no_grad()
    def __call__(self, components, state):
        config, sliding = state.get("prompt_relay_config"), state.get("sliding_window", False)
        if config is None and not sliding:
            return super().__call__(components, state)
        transformer = getattr(components, self.sub_blocks["denoiser"].transformer_name)
        attention_state = prepare_attention_state(
            position_ids=state.get("position_ids"),
            video_indices=state.get("video_indices"),
            text_indices=state.get("text_indices"),
            num_condition_video_rows=state.get("num_condition_video_rows", 0),
            num_latent_frames=state.get("num_latent_frames"),
            duration=state.get("num_frames") / components.fps,
            tokenizer=components.tokenizer,
            config=config,
        )
        window = SlidingWindowConfig(state.get("window_length"), state.get("window_stride")) if sliding else None
        # The processor change is scoped to this invocation, including all steps.
        # No refiner, scheduler, conditioning latent or decoder is replaced.
        with use_relay_attention(transformer, attention_state, window, state.get("query_chunk_size", 128)):
            return super().__call__(components, state)


class RelayDenoiseStep(_RelayLoopMixin, MiniMaxH3DenoiseStep):
    pass


class RelayRef2VADenoiseStep(_RelayLoopMixin, MiniMaxH3Ref2VADenoiseStep):
    pass


def enable_prompt_relay(pipe):
    """Install once AFTER selecting an upstream workflow, BEFORE generating.

    Per-call inputs: prompt_relay_config=None, sliding_window=False,
    window_length=31, window_stride=16, query_chunk_size=128. Supports t2va,
    fl2va and ref2va. Calls to one pipeline must be sequential (as with the
    native mutable schedulers); context parallelism is not supported.
    """
    if not isinstance(pipe, MiniMaxH3ModularPipeline):
        raise TypeError("enable_prompt_relay requires MiniMaxH3ModularPipeline.")
    # `pipe.blocks` returns a deep COPY in the pinned Modular Diffusers API.
    # Work on the execution graph itself; no new model component is registered.
    blocks = pipe._blocks
    if "prompt_relay" in blocks.sub_blocks:
        return pipe
    replacements = {MiniMaxH3DenoiseStep: RelayDenoiseStep, MiniMaxH3Ref2VADenoiseStep: RelayRef2VADenoiseStep}

    def replace_loops(parent):
        count = 0
        for name, block in list(parent.sub_blocks.items()):
            if type(block) in replacements:
                replacement = replacements[type(block)]()
                replacement.sub_blocks = block.sub_blocks
                if hasattr(block, "_progress_bar_config"):
                    replacement._progress_bar_config = block._progress_bar_config.copy()
                parent.sub_blocks[name] = replacement
                count += 1
            elif block.sub_blocks:
                count += replace_loops(block)
        return count

    if not replace_loops(blocks):
        raise ValueError("No native H3 denoising loop found. Use the pinned Diffusers version and an H3 workflow.")
    blocks.sub_blocks.insert("prompt_relay", RelayPromptStep(), 0)
    return pipe
