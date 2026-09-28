# The H3 processor projection/normalization/RoPE sequence is adapted from
# Hugging Face Diffusers, Copyright 2025 The MiniMax Team and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE).
# Unless required by applicable law or agreed to in writing, this code is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Chunked Prompt Relay and overlapping temporal windows for packed H3 Q/K/V."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .schedule import prepare_schedule, token_segment_ids


@dataclass(frozen=True)
class SlidingWindowConfig:
    window_length: int = 31
    window_stride: int = 16

    def __post_init__(self):
        for name in ("window_length", "window_stride"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer in internal frames.")
        if self.window_stride > self.window_length:
            raise ValueError("window_stride must not exceed window_length.")

    def windows(self, frames):
        if type(frames) is not int or frames <= 0:
            raise ValueError("frames must be a positive integer.")
        for start in range(0, frames, self.window_stride):
            end = min(start + self.window_length, frames)
            yield start, end
            if end == frames:
                break


@dataclass
class AttentionState:
    video_rows: torch.Tensor
    frame_ids: torch.Tensor
    other_rows: torch.Tensor
    key_segment_ids: torch.Tensor
    schedule: object
    num_frames: int
    sequence_length: int


def prepare_attention_state(*, position_ids, video_indices, text_indices, num_condition_video_rows,
                            num_latent_frames, duration, tokenizer=None, config=None):
    """Use packed indices, never modality tags (Qwen vision rows also have tag 0)."""
    positions = position_ids.detach().cpu().double()
    video_rows = video_indices.detach().cpu()[num_condition_video_rows:]
    text_rows = text_indices.detach().cpu()
    length = positions.shape[0]
    if video_rows.numel() == 0:
        raise ValueError("H3 layout has no generated video rows.")
    times, frame_ids = torch.unique_consecutive(positions[video_rows, 0], return_inverse=True)
    if len(times) != num_latent_frames or torch.any(times[1:] <= times[:-1]):
        raise ValueError("H3 generated video rows do not match the temporal grid.")
    times = (times - times[0]) / 40.0
    other = torch.ones(length, dtype=torch.bool)
    other[video_rows] = False
    key_segments = torch.full((length,), -1, dtype=torch.long)
    schedule = None
    if config is not None:
        schedule = prepare_schedule(config, times, duration)
        key_segments[text_rows] = token_segment_ids(tokenizer, config, len(text_rows))
        if not schedule.intervals:
            schedule = None
    return AttentionState(video_rows, frame_ids, other.nonzero().flatten(), key_segments,
                          schedule, len(times), length)


def relay_attention(query, key, value, state, window=None, query_chunk_size=128):
    """Video queries are windowed; all other queries see the full sequence once.

    All text, reference/keyframe rows and audio remain globally visible. Only
    generated-video-query/local-text-key logits receive the Gaussian tail cost,
    following Hunyuan's video-query-only rule. Global RoPE is applied by the
    processor BEFORE slicing windows. Overlap outputs are averaged in FP32.
    """
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Expected matching [B, L, H, D] Q/K/V tensors.")
    if query.shape[0] != 1 or query.shape[1] != state.sequence_length:
        raise ValueError("H3 relay requires one complete, unsharded packed request.")
    if type(query_chunk_size) is not int or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer.")
    device = query.device
    video_rows, frame_ids, other_rows, key_segments = (
        tensor.to(device) for tensor in (state.video_rows, state.frame_ids, state.other_rows, state.key_segment_ids)
    )
    q, k, v = (tensor.transpose(1, 2) for tensor in (query, key, value))
    output = torch.empty_like(query)
    accum_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    accumulated = torch.zeros_like(query[:, video_rows], dtype=accum_dtype)
    coverage = torch.zeros(len(video_rows), device=device, dtype=accum_dtype)
    schedule = state.schedule
    if schedule is not None:
        coordinates = schedule.coordinates.to(device)
        midpoints = schedule.midpoints.to(device)
        widths = schedule.half_widths.to(device)

    windows = window.windows(state.num_frames) if window else [(0, state.num_frames)]
    for start, end in windows:
        selected = ((frame_ids >= start) & (frame_ids < end)).nonzero().flatten()
        rows = video_rows[selected]
        # Keep original packed key order, including in the full-length window.
        keys = torch.cat((rows, other_rows)).sort().values
        window_k, window_v = k[:, :, keys], v[:, :, keys]
        if schedule is not None:
            local_keys = (key_segments[keys] >= 0).nonzero().flatten()
            segments = key_segments[keys[local_keys]]
        for offset in range(0, len(rows), query_chunk_size):
            indices = selected[offset:offset + query_chunk_size]
            chunk_rows = video_rows[indices]
            bias = None
            if schedule is not None:
                distance = (coordinates[frame_ids[indices], None] - midpoints[None]).abs()
                cost = torch.relu(distance - widths[None]).square() / (2 * schedule.sigma**2)
                bias = torch.zeros(len(indices), len(keys), device=device, dtype=query.dtype)
                bias[:, local_keys] = -cost[:, segments].clamp(max=torch.finfo(query.dtype).max).to(query.dtype)
                bias = bias[None, None]
            chunk = F.scaled_dot_product_attention(q[:, :, chunk_rows], window_k, window_v,
                                                   attn_mask=bias, dropout_p=0.0, is_causal=False)
            accumulated[:, indices] += chunk.transpose(1, 2).to(accum_dtype)
        coverage[selected] += 1
    output[:, video_rows] = (accumulated / coverage[None, :, None, None]).to(query.dtype)
    for offset in range(0, len(other_rows), query_chunk_size):
        rows = other_rows[offset:offset + query_chunk_size]
        output[:, rows] = F.scaled_dot_product_attention(q[:, :, rows], k, v, dropout_p=0.0,
                                                        is_causal=False).transpose(1, 2)
    return output


class MiniMaxH3RelayAttnProcessor:
    def __init__(self, state, window=None, query_chunk_size=128):
        self.state, self.window, self.query_chunk_size = state, window, query_chunk_size

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        if attention_mask is not None:
            raise ValueError("H3 relay expects the native unpadded packed sequence without an external mask.")
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        hidden_states = relay_attention(q, k, v, self.state, self.window, self.query_chunk_size)
        hidden_states = hidden_states.flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](hidden_states))


@contextmanager
def use_relay_attention(transformer, state, window=None, query_chunk_size=128):
    """Replace only joint transformer attention; restore even after an exception."""
    if state.schedule is None and window is None:
        yield
        return
    attentions = [block.attn for block in transformer.transformer_blocks]
    original = [attn.processor for attn in attentions]
    if any(getattr(p, "_parallel_config", None) is not None for p in original):
        raise ValueError("H3 relay currently requires unsharded attention; disable context parallelism.")
    try:
        for attn in attentions:
            attn.set_processor(MiniMaxH3RelayAttnProcessor(state, window, query_chunk_size))
        yield
    finally:
        for attn, processor in zip(attentions, original):
            attn.set_processor(processor)
