# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

from ..visual_relay import shifted_rope_positions

__all__ = [
    'flash_attention',
    'attention',
    'visual_relay_shifted_attention',
]


def _shift_temporal_rope(x, position_delta, freqs):
    """Shift only temporal RoPE on an already-rotated token tensor."""
    if position_delta == 0:
        return x
    original_dtype = x.dtype
    complex_dim = x.size(-1) // 2
    temporal_dim = complex_dim - 2 * (complex_dim // 3)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], complex_dim, 2))
    base_phase = torch.angle(freqs[1, :temporal_dim]).to(
        device=x.device, dtype=torch.float32)
    multiplier = torch.polar(
        torch.ones_like(base_phase),
        float(position_delta) * base_phase,
    ).to(x_complex.dtype).view(*([1] * (x.ndim - 1)), temporal_dim)
    shifted_temporal = x_complex[..., :temporal_dim] * multiplier
    shifted = torch.cat(
        [shifted_temporal, x_complex[..., temporal_dim:]], dim=-1)
    return torch.view_as_real(shifted).flatten(-2).to(original_dtype)


def visual_relay_shifted_attention(
    q,
    k,
    v,
    seq_lens,
    grid_sizes,
    anchor_indices,
    freqs,
    anchor_rope_offset=0.5,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """Apply Visual Relay's shifted-anchor RoPE to one expanded J stream.

    Ordinary frames retain their final-output temporal coordinates after all
    temporary anchors are removed.  An anchor inserted immediately before
    output frame ``p`` receives coordinate ``p - 1 + anchor_rope_offset``;
    offset 0.5 therefore places it halfway between output frames p-1 and p.
    Ordinary queries remain bidirectional, while every anchor query reads only
    its own anchor and the suffix, matching the working two-image baseline.
    """
    anchors = tuple(int(index) for index in anchor_indices)
    if q.size(0) != 1 or k.size(0) != 1 or v.size(0) != 1:
        raise ValueError("shifted Visual Relay attention expects one J stream")
    if window_size != (-1, -1):
        raise ValueError("shifted Visual Relay requires global self-attention")
    if not anchors or any(left >= right for left, right in zip(anchors, anchors[1:])):
        raise ValueError("Visual Relay anchor indices must be strictly increasing")
    if not 0.0 <= float(anchor_rope_offset) <= 1.0:
        raise ValueError("anchor_rope_offset must be in [0, 1]")

    frames, height, width = (
        int(value) for value in grid_sizes[0].tolist())
    tokens_per_frame = height * width
    sequence_length = frames * tokens_per_frame
    if int(seq_lens[0]) != sequence_length:
        raise ValueError("Visual Relay sequence length disagrees with its grid")
    if anchors[0] <= 0 or anchors[-1] >= frames:
        raise ValueError("Visual Relay anchors must be interior latent frames")

    shifted_q = q.clone()
    shifted_k = k.clone()
    positions = shifted_rope_positions(
        frames, anchors, anchor_rope_offset=anchor_rope_offset
    )
    for frame, position in enumerate(positions):
        delta = position - frame
        if delta == 0:
            continue
        start = frame * tokens_per_frame
        end = start + tokens_per_frame
        shifted_q[:, start:end] = _shift_temporal_rope(
            shifted_q[:, start:end], delta, freqs)
        shifted_k[:, start:end] = _shift_temporal_rope(
            shifted_k[:, start:end], delta, freqs)

    output = flash_attention(
        q=shifted_q,
        k=shifted_k,
        v=v,
        k_lens=seq_lens,
        window_size=window_size,
        dtype=dtype,
    )
    # The working Visual Relay path deliberately keeps native relative RoPE
    # for an anchor's own one-way self/suffix attention.
    for anchor in anchors:
        start = anchor * tokens_per_frame
        end = start + tokens_per_frame
        output[:, start:end] = flash_attention(
            q=q[:, start:end],
            k=k[:, start:sequence_length],
            v=v[:, start:sequence_length],
            window_size=window_size,
            dtype=dtype,
        )
    return output


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
