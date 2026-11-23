"""Wan-equivalent Prompt Relay over MiniMax H3 joint attention."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import PromptRelayConfig, ResolvedPromptRelaySchedule, resolve_schedule


@dataclass(frozen=True)
class AttentionState:
    video_rows: torch.Tensor
    frame_ids: torch.Tensor
    other_rows: torch.Tensor
    key_segment_ids: torch.Tensor
    schedule: ResolvedPromptRelaySchedule
    sequence_length: int


def token_segment_ids(
    tokenizer,
    config: PromptRelayConfig,
    num_text_tokens: int,
) -> torch.Tensor:
    """Map H3 text rows to local-prompt IDs; global/media rows remain ``-1``."""

    text, spans = config.full_prompt_and_spans
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    input_ids = encoded["input_ids"]
    if len(offsets) != len(input_ids) or num_text_tokens < len(offsets):
        raise ValueError(
            "H3 prompt embeddings do not match the complete, untruncated prompt"
        )

    prefix_length = num_text_tokens - len(offsets)
    segment_ids = torch.full((num_text_tokens,), -1, dtype=torch.long)
    for token_index, (token_start, token_end) in enumerate(offsets):
        matches = [
            index
            for index, (span_start, span_end) in enumerate(spans)
            if token_start < span_end and token_end > span_start
        ]
        if len(matches) > 1:
            raise ValueError(
                "A tokenizer token crosses two local prompts; revise the prompt "
                "boundary"
            )
        if not matches:
            continue

        segment_index = matches[0]
        span_start, span_end = spans[segment_index]
        outside = (
            text[token_start:min(token_end, span_start)]
            + text[max(token_start, span_end):token_end]
        )
        if outside.strip():
            raise ValueError(
                "A tokenizer token crosses global and local prompt content; "
                "revise the prompt boundary"
            )
        segment_ids[prefix_length + token_index] = segment_index

    for index in range(len(spans)):
        if not torch.any(segment_ids == index):
            raise ValueError(f"local_prompts[{index}] has no encoded tokens")
    return segment_ids


def prepare_attention_state(
    *,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    text_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_latent_frames: int,
    tokenizer,
    config: PromptRelayConfig,
) -> AttentionState:
    """Resolve H3 packed rows and construct a Wan-compatible relay schedule."""

    positions = position_ids.detach().cpu()
    video_rows = video_indices.detach().cpu()[num_condition_video_rows:]
    text_rows = text_indices.detach().cpu()
    sequence_length = positions.shape[0]
    if video_rows.numel() == 0:
        raise ValueError("H3 layout has no generated-video rows")

    times = positions[video_rows, 0]
    _, frame_ids = torch.unique_consecutive(times, return_inverse=True)
    if int(frame_ids.max()) + 1 != num_latent_frames:
        raise ValueError(
            "H3 generated-video rows do not match the internal-frame grid"
        )

    other_mask = torch.ones(sequence_length, dtype=torch.bool)
    other_mask[video_rows] = False
    key_segment_ids = torch.full((sequence_length,), -1, dtype=torch.long)
    key_segment_ids[text_rows] = token_segment_ids(
        tokenizer, config, len(text_rows)
    )
    return AttentionState(
        video_rows=video_rows,
        frame_ids=frame_ids,
        other_rows=other_mask.nonzero().flatten(),
        key_segment_ids=key_segment_ids,
        schedule=resolve_schedule(config, num_latent_frames),
        sequence_length=sequence_length,
    )


def _segment_costs(
    frame_ids: torch.Tensor,
    schedule: ResolvedPromptRelaySchedule,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return float32 costs shaped ``[video queries, local prompts]``."""

    frames = frame_ids.to(device=device, dtype=torch.float32)[:, None]
    starts = torch.tensor(
        [start for start, _ in schedule.intervals],
        device=device,
        dtype=torch.float32,
    )[None]
    ends = torch.tensor(
        [end for _, end in schedule.intervals],
        device=device,
        dtype=torch.float32,
    )[None]

    if schedule.mode == "hard":
        inside = (frames >= starts) & (frames < ends)
        return torch.where(
            inside,
            torch.zeros_like(frames + starts),
            torch.full_like(frames + starts, 1e9),
        )

    midpoints = torch.tensor(
        schedule.midpoints, device=device, dtype=torch.float32
    )[None]
    windows = torch.tensor(
        schedule.inner_windows, device=device, dtype=torch.float32
    )[None]
    distance = (frames - midpoints).abs()
    return torch.relu(distance - windows).square() / (2.0 * schedule.sigma**2)


def relay_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: AttentionState,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """Apply Prompt Relay only to generated-video-query/local-text-key logits."""

    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Expected matching [B, L, H, D] Q/K/V tensors")
    if query.shape[0] != 1 or query.shape[1] != state.sequence_length:
        raise ValueError("H3 Prompt Relay requires one complete packed request")
    if type(query_chunk_size) is not int or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer")

    device = query.device
    video_rows = state.video_rows.to(device)
    frame_ids = state.frame_ids.to(device)
    other_rows = state.other_rows.to(device)
    key_segment_ids = state.key_segment_ids.to(device)
    local_keys = (key_segment_ids >= 0).nonzero().flatten()
    local_key_segments = key_segment_ids[local_keys]
    costs = _segment_costs(frame_ids, state.schedule, device=device)

    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    output = torch.empty_like(query)
    max_bias = torch.finfo(query.dtype).max

    for offset in range(0, len(video_rows), query_chunk_size):
        selected = slice(offset, offset + query_chunk_size)
        rows = video_rows[selected]
        bias = torch.zeros(
            len(rows), state.sequence_length, device=device, dtype=query.dtype
        )
        selected_costs = costs[selected][:, local_key_segments]
        bias[:, local_keys] = -selected_costs.clamp(max=max_bias).to(query.dtype)
        chunk = F.scaled_dot_product_attention(
            q[:, :, rows],
            k,
            v,
            attn_mask=bias[None, None],
            dropout_p=0.0,
            is_causal=False,
        )
        output[:, rows] = chunk.transpose(1, 2)

    for offset in range(0, len(other_rows), query_chunk_size):
        rows = other_rows[offset:offset + query_chunk_size]
        chunk = F.scaled_dot_product_attention(
            q[:, :, rows], k, v, dropout_p=0.0, is_causal=False
        )
        output[:, rows] = chunk.transpose(1, 2)
    return output


class MiniMaxH3PromptRelayAttnProcessor:
    """Diffusers attention processor for H3 transformer blocks."""

    def __init__(self, state: AttentionState, query_chunk_size: int = 128):
        self.state = state
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        attn,
        hidden_states,
        rotary_emb=None,
        attention_mask=None,
    ):
        from diffusers.models.transformers.transformer_minimax_h3 import (
            _apply_rotary_emb,
        )

        if attention_mask is not None:
            raise ValueError(
                "H3 Prompt Relay expects the native unpadded packed sequence"
            )
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query, key, value = (
            tensor.unflatten(-1, (attn.heads, -1))
            for tensor in (query, key, value)
        )
        query, key = attn.norm_q(query), attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        hidden_states = relay_attention(
            query, key, value, self.state, self.query_chunk_size
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


@contextmanager
def use_prompt_relay_attention(
    transformer,
    state: AttentionState,
    query_chunk_size: int = 128,
):
    """Install relay processors for one denoising invocation, then restore them."""

    attentions = [block.attn for block in transformer.transformer_blocks]
    original_processors = [attention.processor for attention in attentions]
    if any(
        getattr(processor, "_parallel_config", None) is not None
        for processor in original_processors
    ):
        raise ValueError(
            "MiniMax H3 Prompt Relay currently requires unsharded attention; "
            "disable context parallelism"
        )
    try:
        for attention in attentions:
            attention.set_processor(
                MiniMaxH3PromptRelayAttnProcessor(state, query_chunk_size)
            )
        yield
    finally:
        for attention, processor in zip(attentions, original_processors):
            attention.set_processor(processor)
