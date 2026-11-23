"""Prompt Relay blocks for the official MiniMax H3 modular pipeline."""

from __future__ import annotations

import inspect

import torch
from PIL import Image, ImageOps
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
)
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


class IndexedKeyframeStep(ModularPipelineBlocks):
    """Replace native first/last keyframes with indexed latent-frame anchors."""

    model_name = "minimax-h3"

    @property
    def description(self):
        return "Prepare arbitrary indexed FL2VA keyframes on the target canvas."

    @property
    def inputs(self):
        return [
            InputParam("indexed_keyframes", type_hint=list, default=None),
            InputParam("height", type_hint=int, default=None),
            InputParam("width", type_hint=int, default=None),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam("keyframes", type_hint=list),
            OutputParam("keyframe_anchors", type_hint=tuple),
        ]

    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        indexed = block_state.indexed_keyframes
        if indexed is None:
            return components, state
        if not indexed:
            raise ValueError("indexed_keyframes must not be empty")
        if block_state.height is None or block_state.width is None:
            raise ValueError("Indexed keyframes require a resolved target canvas")

        size = (block_state.width, block_state.height)
        keyframes, anchors = [], []
        for image, anchor in indexed:
            if type(anchor) is not int or anchor < 0:
                raise ValueError("Indexed keyframe anchors must be non-negative integers")
            image = image.convert("RGB")
            keyframes.append(
                ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
            )
            anchors.append(anchor)
        if len(set(anchors)) != len(anchors):
            raise ValueError("Indexed keyframe anchors must be unique")

        block_state.keyframes = keyframes
        block_state.keyframe_anchors = tuple(anchors)
        self.set_block_state(state, block_state)
        return components, state


class IndexedMiniMaxH3PrepareLayoutStep(MiniMaxH3PrepareLayoutStep):
    """Allow integer keyframe anchors in the native H3 packed layout."""

    @staticmethod
    def build_packed_sequence(*args, **kwargs):
        native_builder = MiniMaxH3PrepareLayoutStep.build_packed_sequence
        bound = inspect.signature(native_builder).bind_partial(*args, **kwargs)
        anchors = bound.arguments.get("keyframe_anchors")
        anchors = tuple(anchors or ())
        native_anchors = tuple(
            anchor if anchor in ("first", "last") else "first"
            for anchor in anchors
        )
        bound.arguments["keyframe_anchors"] = native_anchors
        result = list(native_builder(*bound.args, **bound.kwargs))
        if not any(type(anchor) is int for anchor in anchors):
            return tuple(result)

        text_token_tags = bound.arguments["text_token_tags"]
        num_latent_frames = bound.arguments["num_latent_frames"]
        latent_height = bound.arguments["latent_height"]
        latent_width = bound.arguments["latent_width"]
        patch_size = bound.arguments["patch_size"]
        _, patch_h, patch_w = patch_size
        rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        condition_start = len(text_token_tags)
        target_start = result[0].shape[0] - num_latent_frames * rows_per_frame

        for index, anchor in enumerate(anchors):
            if type(anchor) is not int:
                continue
            if not 0 <= anchor < num_latent_frames:
                raise ValueError(
                    f"Keyframe anchor {anchor} is outside the internal frame "
                    f"range [0, {num_latent_frames})"
                )
            condition_rows = slice(
                condition_start + index * rows_per_frame,
                condition_start + (index + 1) * rows_per_frame,
            )
            target_row = target_start + anchor * rows_per_frame
            result[0][condition_rows, 0] = result[0][target_row, 0]
        return tuple(result)


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

    def replace_layouts(parent):
        for name, block in list(parent.sub_blocks.items()):
            if type(block) is MiniMaxH3PrepareLayoutStep:
                replacement = IndexedMiniMaxH3PrepareLayoutStep()
                if hasattr(block, "_progress_bar_config"):
                    replacement._progress_bar_config = (
                        block._progress_bar_config.copy()
                    )
                parent.sub_blocks[name] = replacement
            elif block.sub_blocks:
                replace_layouts(block)

    replace_layouts(blocks)
    blocks.sub_blocks.insert("prompt_relay", PromptRelayInputStep(), 0)
    before_encode_index = list(blocks.sub_blocks).index("before_encode")
    blocks.sub_blocks.insert(
        "indexed_keyframes", IndexedKeyframeStep(), before_encode_index + 1
    )
    return pipe
