"""Prompt Relay blocks for the official MiniMax H3 modular pipeline."""

from __future__ import annotations

import torch
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3DenoiseStep,
    MiniMaxH3Ref2VADenoiseStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MiniMaxH3ModularPipeline,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam

from .attention import prepare_attention_state, use_prompt_relay_attention
from .config import PromptRelayConfig


class PromptRelayInputStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self):
        return "Build Wan-compatible Prompt Relay text before H3 encoding."

    @property
    def inputs(self):
        return [
            InputParam("prompt", type_hint=str, default=None),
            InputParam("prompt_relay_config", type_hint=dict, default=None),
            InputParam("query_chunk_size", type_hint=int, default=128),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("prompt", type_hint=str)]

    def __call__(self, components, state):
        chunk_size = state.get("query_chunk_size")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")

        raw_config = state.get("prompt_relay_config")
        prompt = state.get("prompt")
        if raw_config is not None:
            if prompt not in (None, ""):
                raise ValueError("Provide prompt_relay_config or prompt, not both")
            config = (
                raw_config
                if isinstance(raw_config, PromptRelayConfig)
                else PromptRelayConfig.from_mapping(raw_config)
            )
            state.set("prompt_relay_config", config)
            state.set("prompt", config.full_prompt)
            prompt = config.full_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a non-empty prompt or prompt_relay_config")
        return components, state


class _PromptRelayLoopMixin:
    @torch.no_grad()
    def __call__(self, components, state):
        config = state.get("prompt_relay_config")
        if config is None:
            return super().__call__(components, state)
        if not isinstance(config, PromptRelayConfig):
            config = PromptRelayConfig.from_mapping(config)

        transformer = getattr(
            components, self.sub_blocks["denoiser"].transformer_name
        )
        attention_state = prepare_attention_state(
            position_ids=state.get("position_ids"),
            video_indices=state.get("video_indices"),
            text_indices=state.get("text_indices"),
            num_condition_video_rows=state.get("num_condition_video_rows", 0),
            num_latent_frames=state.get("num_latent_frames"),
            tokenizer=components.tokenizer,
            config=config,
        )
        with use_prompt_relay_attention(
            transformer,
            attention_state,
            state.get("query_chunk_size", 128),
        ):
            return super().__call__(components, state)


class PromptRelayDenoiseStep(_PromptRelayLoopMixin, MiniMaxH3DenoiseStep):
    pass


class PromptRelayRef2VADenoiseStep(
    _PromptRelayLoopMixin, MiniMaxH3Ref2VADenoiseStep
):
    pass


def enable_prompt_relay(pipe):
    """Install Prompt Relay after selecting an H3 workflow and before inference."""

    if not isinstance(pipe, MiniMaxH3ModularPipeline):
        raise TypeError("enable_prompt_relay requires MiniMaxH3ModularPipeline")
    blocks = pipe._blocks
    if "prompt_relay" in blocks.sub_blocks:
        return pipe

    replacements = {
        MiniMaxH3DenoiseStep: PromptRelayDenoiseStep,
        MiniMaxH3Ref2VADenoiseStep: PromptRelayRef2VADenoiseStep,
    }

    def replace_loops(parent):
        count = 0
        for name, block in list(parent.sub_blocks.items()):
            if type(block) in replacements:
                replacement = replacements[type(block)]()
                replacement.sub_blocks = block.sub_blocks
                if hasattr(block, "_progress_bar_config"):
                    replacement._progress_bar_config = (
                        block._progress_bar_config.copy()
                    )
                parent.sub_blocks[name] = replacement
                count += 1
            elif block.sub_blocks:
                count += replace_loops(block)
        return count

    if not replace_loops(blocks):
        raise ValueError(
            "No native H3 denoising loop found; install the Diffusers revision "
            "pinned in requirements.txt"
        )
    blocks.sub_blocks.insert("prompt_relay", PromptRelayInputStep(), 0)
    return pipe
