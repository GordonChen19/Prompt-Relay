"""TS-Attn and SwitchCraft on MiniMax H3's joint attention: the tensor half of the port (see h3_edit_plan.py).

    install(transformer, state)   wraps every transformer block's attention processor and counts the steps;
                                  blocks/steps outside the method's schedule call the original processor
                                  untouched, so they are exactly plain H3.

Inside the schedule the block computes q, k, v exactly as diffusers' MiniMaxH3AttnProcessor does (projection,
per-head RMSNorm, rotary) and then attends with an additive bias B on the (generated-video query, text key)
logits only, inside H3's one softmax over the whole packed sequence:

  SwitchCraft  the authors' own edit (edit_Q_with_rawK_projector, read unchanged from their model.py) turns each
               event's video queries q into q'. In Wan the edited q' attends to the text only; here q' must meet
               the text keys while the video and audio keys still see q, so B = (q' - q) . k_text / sqrt(D):
               the text logits become exactly q' . k_text / sqrt(D), every other logit stays q . k / sqrt(D).
  TS-Attn      the authors' soft mask (TsAttnCrossAttnProcessor.build_cross_attn_mask with the subject region
               of WanTransformerBlock.forward), computed from the video->text logits of this block:
               on the subject region of event e's rows, + alpha*(max - mean) on event e's and the subject's
               tokens, + alpha*(min - mean) on the other events' tokens, alpha = 1 + 0.5*(1 - normalised prob).
               As released, event k's max/min/mean/alpha come from the rows starting at row k (their start_idx
               advances by 1); ts_stats="own" takes each row's own (see ts_stat_rows).

Memory stays bounded: video queries go in chunks, each with a [heads, chunk, sequence] bias (as Gordon's H3
Prompt Relay does), so the 33B model plus the edit fits one 80 GB GPU with CPU offload.
"""
import ast
import contextlib
import math
import os

import torch
import torch.nn.functional as F

from h3_edit_plan import (SWITCHCRAFT, TS_ATTN, TS_ALPHA_MAX, TS_ALPHA_MIN, TS_TOP_FRACTION, PlanError,
                          ts_event_counts)

SC_NESTED = ("_no_amp_ctx", "_build_right_projector_from_K", "_basis_from_K_per_head",
             "solve_abss_batched_full", "edit_Q_with_rawK_projector")


# ---------------------------------------------------------------- the authors' SwitchCraft edit, unchanged
def class_method_node(tree, cls, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
    raise KeyError("%s.%s not found" % (cls, name))


def switchcraft_model_py(switchcraft_dir):
    return os.path.join(os.path.abspath(os.path.expanduser(switchcraft_dir)), "wan", "modules", "model.py")


def load_switchcraft_edit(switchcraft_dir):
    """edit_Q_with_rawK_projector and its helpers, the functions defined inside the authors'
    WanT2VCrossAttention.forward, compiled from their file as they are. Returns (function, {name: source})."""
    path = switchcraft_model_py(switchcraft_dir)
    if not os.path.isfile(path):
        raise SystemExit("SwitchCraft code not found: %s" % path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    fwd = class_method_node(tree, "WanT2VCrossAttention", "forward")
    defs = {n.name: n for n in fwd.body if isinstance(n, ast.FunctionDef)}
    missing = [n for n in SC_NESTED if n not in defs]
    if missing:
        raise SystemExit("SwitchCraft's forward no longer defines %s (%s)" % (missing, path))
    body = [defs[n] for n in SC_NESTED]
    scope = {"torch": torch, "F": F, "math": math, "nullcontext": contextlib.nullcontext,
             "__name__": "switchcraft_author_edit"}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), scope)
    sources = {n: ast.get_source_segment(src, defs[n]) for n in SC_NESTED}
    return scope["edit_Q_with_rawK_projector"], sources


# ---------------------------------------------------------------- where things are in H3's packed sequence
class Layout:
    """Rows of one t2va packed sequence: text rows, generated video rows (frame by frame, each frame in (h, w)
    raster order) and the rest; read from the transformer's own position_ids / video_indices / text_indices."""

    def __init__(self, position_ids, video_indices, text_indices):
        pos = position_ids.detach()
        vid = video_indices.detach().long()
        txt = text_indices.detach().long()
        if pos.ndim != 2 or pos.shape[1] != 3 or vid.numel() == 0 or txt.numel() == 0:
            raise PlanError("unexpected H3 layout")
        times, frame_ids = torch.unique_consecutive(pos[vid, 0], return_inverse=True)
        n_frames = times.numel()
        if n_frames > 1 and not bool((times[1:] > times[:-1]).all()):
            raise PlanError("video rows are not frame after frame (only t2va, no keyframes, is supported)")
        counts = torch.bincount(frame_ids, minlength=n_frames)
        per_frame = int(counts[0])
        if not bool((counts == per_frame).all()):
            raise PlanError("video frames have different row counts (only t2va is supported)")
        hw = pos[vid, 1:].reshape(n_frames, per_frame, 2)
        first = hw[0]
        h_vals, w_vals = torch.unique(first[:, 0]), torch.unique(first[:, 1])
        hp, wp = h_vals.numel(), w_vals.numel()
        r = torch.arange(per_frame, device=pos.device)
        h_rank = torch.searchsorted(h_vals, first[:, 0].contiguous())
        w_rank = torch.searchsorted(w_vals, first[:, 1].contiguous())
        if (hp * wp != per_frame or not bool((h_rank == r // wp).all()) or not bool((w_rank == r % wp).all())
                or not bool((hw == first[None]).all())):
            raise PlanError("video rows are not an (h, w) raster repeated every frame")
        other = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        other[vid] = False
        self.video_rows, self.text_rows = vid, txt
        self.other_rows = other.nonzero().flatten()
        self.frames, self.hp, self.wp = n_frames, hp, wp
        self.n_video, self.n_text, self.seq_len = vid.numel(), txt.numel(), pos.shape[0]
        a = int(txt[0])
        self.text_slice = slice(a, a + txt.numel()) if bool((txt == torch.arange(
            a, a + txt.numel(), device=txt.device)).all()) else None

    def to(self, device):
        for name in ("video_rows", "text_rows", "other_rows"):
            setattr(self, name, getattr(self, name).to(device))
        return self


# ---------------------------------------------------------------- attention with a bias on video->text logits
def biased_attention(query, key, value, layout, bias_for_chunk, chunk_size):
    """softmax(q k^T / sqrt(D) + B) v for every row of the packed sequence ([1, L, H, D] in and out).
    B is zero except on (generated-video query, text key); bias_for_chunk(s, e) gives it for video rows
    s..e-1 (in video-row order) as [H, e - s, n_text], or None for no bias on that chunk."""
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    out = torch.empty_like(query)
    heads, length = query.shape[2], key.shape[1]
    vid, other = layout.video_rows, layout.other_rows
    for s in range(0, vid.numel(), chunk_size):
        e = min(s + chunk_size, vid.numel())
        rows = vid[s:e]
        b = bias_for_chunk(s, e)
        if b is None:
            o = F.scaled_dot_product_attention(q[:, :, rows], k, v, dropout_p=0.0, is_causal=False)
        else:
            mask = query.new_zeros(1, heads, e - s, length)
            if layout.text_slice is not None:
                mask[0, :, :, layout.text_slice] = b.to(query.dtype)
            else:
                mask[0][:, :, layout.text_rows] = b.to(query.dtype)
            o = F.scaled_dot_product_attention(q[:, :, rows], k, v, attn_mask=mask, dropout_p=0.0,
                                               is_causal=False)
        out[:, rows] = o.transpose(1, 2)
    for s in range(0, other.numel(), chunk_size):
        rows = other[s:s + chunk_size]
        o = F.scaled_dot_product_attention(q[:, :, rows], k, v, dropout_p=0.0, is_causal=False)
        out[:, rows] = o.transpose(1, 2)
    return out


# ---------------------------------------------------------------- SwitchCraft
def switchcraft_bias(query, key, layout, edit_fn, event_cols, shares):
    """B = (q' - q) . k_text / sqrt(D) for every generated-video row, q' = the authors' edit of q."""
    qv = query[:, layout.video_rows]                       # [1, Nv, H, D]
    kt = key[:, layout.text_rows]                          # [1, M, H, D]
    grid = torch.tensor([[layout.frames, layout.hp, layout.wp]], dtype=torch.long)
    q_edit = edit_fn(qv.clone(), kt, grid, action_cols=[list(c) for c in event_cols],
                     event_len_ratio=list(shares), renorm=True)
    dq = (q_edit.float() - qv.float())[0].transpose(0, 1)  # [H, Nv, D]
    kt_t = kt[0].float().permute(1, 2, 0)                  # [H, D, M]
    scale = 1.0 / math.sqrt(query.shape[-1])

    def bias(s, e):
        return torch.matmul(dq[:, s:e], kt_t) * scale       # [H, e - s, M]
    return bias


# ---------------------------------------------------------------- TS-Attn
def ts_alpha(sim, alpha_min=TS_ALPHA_MIN, alpha_max=TS_ALPHA_MAX, eps=1e-12):
    """Their get_dynamic_alpha: 1 for the most-attended token of each query/head, up to 1.5 for the least."""
    p = F.softmax(sim, dim=-1)
    p_min = p.min(dim=-1, keepdim=True).values
    p_max = p.max(dim=-1, keepdim=True).values
    p = (p - p_min) / (p_max - p_min + eps)
    return alpha_min + (1 - p) * (alpha_max - alpha_min)


def erode_isolated(mask_1d, frames, hp, wp, k=3):
    """Their erode_isolated: keep a region pixel only if one of its 8 neighbours (same frame) is in it."""
    grid = mask_1d.reshape(frames, 1, hp, wp).float()
    kernel = torch.ones((1, 1, k, k), device=grid.device, dtype=grid.dtype)
    kernel[0, 0, k // 2, k // 2] = 0
    neighbours = F.conv2d(grid, kernel, stride=1, padding=k // 2)
    return ((neighbours > 0) & (grid > 0)).reshape(-1)


def ts_event_ids(n_rows, shares, device):
    counts = ts_event_counts(n_rows, list(shares))
    return torch.repeat_interleave(torch.arange(len(counts), device=device),
                                   torch.tensor(counts, device=device))


def ts_subject_region(logits_fn, n_video, n_text, subject_cols, layout, chunk_size, top_fraction=TS_TOP_FRACTION):
    """Their subject layout (WanTransformerBlock.forward): softmax over the text, mean over heads, min-max
    normalised over all video rows and tokens, mean over the subject's tokens, kept above the mean of the top
    half, then eroded. Returns a bool per generated-video row."""
    attn_map = torch.empty(n_video, n_text, dtype=torch.float32, device=layout.video_rows.device)
    for s in range(0, n_video, chunk_size):
        e = min(s + chunk_size, n_video)
        attn_map[s:e] = F.softmax(logits_fn(s, e), dim=-1).mean(dim=0)
    amin, amax = attn_map.amin(), attn_map.amax()
    normalized = (attn_map - amin) / (amax - amin + 1e-6)
    cols = torch.as_tensor(subject_cols, dtype=torch.long, device=attn_map.device)
    selected = normalized[:, cols].mean(dim=-1)
    ranked = torch.sort(selected, descending=True).values
    top_mean = ranked[: int(n_video * top_fraction)].mean()
    region = selected > top_mean
    return erode_isolated(region, layout.frames, layout.hp, layout.wp)


def ts_stat_rows(events, starts, stats):
    """Which video row's logits give the statistics (max, min, mean, alpha) of each row.
    'released': as their build_cross_attn_mask computes them - it advances start_idx by
    subject_visual_mask.size(1) after unsqueezing it, i.e. by 1, so event k's local row t reads row k + t.
    'own': each row its own logits (what the paper describes)."""
    n = events.numel()
    rows = torch.arange(n, device=events.device)
    if stats == "own":
        return rows
    return events + (rows - starts[events])


def ts_attn_bias(query, key, layout, event_cols, subject_cols, shares, chunk_size, raw=None, stats="released"):
    """The authors' soft mask for every generated-video row, chunk by chunk ([H, c, M]).
    Logits: this block's video->text attention logits (raw=(q, k) before norm and rotary: their literal
    to_q / to_k choice). Only the last subject counts, as in their build_cross_attn_mask. stats: see
    ts_stat_rows."""
    src_q, src_k = (query, key) if raw is None else raw
    qv = src_q[0, layout.video_rows].transpose(0, 1)                 # [H, Nv, D]
    kt = src_k[0, layout.text_rows].float().permute(1, 2, 0)         # [H, D, M]
    n_video, n_text = qv.shape[1], kt.shape[2]
    scale = 1.0 / math.sqrt(query.shape[-1])

    def logits(s, e):
        return torch.matmul(qv[:, s:e].float(), kt) * scale            # [H, c, M]

    region = ts_subject_region(logits, n_video, n_text, subject_cols[-1], layout, chunk_size)
    events = ts_event_ids(n_video, shares, qv.device)
    dev = qv.device
    counts = torch.bincount(events, minlength=len(event_cols))
    starts = torch.cumsum(counts, 0) - counts
    stat_rows = ts_stat_rows(events, starts, stats)
    if int(stat_rows.min()) < 0 or int(stat_rows.max()) >= n_video:
        raise PlanError("TS-Attn statistics rows fall outside the video")
    ev_masks = torch.zeros(len(event_cols), n_text, dtype=torch.bool, device=dev)
    for a, cols in enumerate(event_cols):
        ev_masks[a, torch.as_tensor(cols, dtype=torch.long, device=dev)] = True
    subj = torch.zeros(n_text, dtype=torch.bool, device=dev)
    subj[torch.as_tensor(subject_cols[-1], dtype=torch.long, device=dev)] = True
    positive = (ev_masks | subj).float()                              # own event's tokens + subject
    negative = ev_masks.float().sum(0, keepdim=True) - ev_masks.float()   # every other event's tokens

    def bias(s, e):
        keep = region[s:e]
        if not bool(keep.any()):
            return None
        rows = stat_rows[s:e]
        sim = torch.matmul(qv[:, rows].float(), kt) * scale                # [H, c, M] of the statistics rows
        top, low, mean = sim.amax(-1, keepdim=True), sim.amin(-1, keepdim=True), sim.mean(-1, keepdim=True)
        ev = events[s:e]
        b = ts_alpha(sim) * ((top - mean) * positive[ev][None] + (low - mean) * negative[ev][None])
        return b * keep.float()[None, :, None]
    return bias


# ---------------------------------------------------------------- the switch inside H3
def diffusers_rope():
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    return _apply_rotary_emb


class EditState:
    """What the wrapped processors read: the method, its schedule, this video's plan and the step counter."""

    def __init__(self, method, until_step, blocks, chunk_size=128, edit_fn=None, ts_logits="attn",
                 ts_stats="released"):
        if method not in (SWITCHCRAFT, TS_ATTN):
            raise PlanError("unknown method %r" % method)
        if method == SWITCHCRAFT and edit_fn is None:
            raise PlanError("SwitchCraft needs the authors' edit function")
        if ts_logits not in ("attn", "raw"):
            raise PlanError("ts_logits must be 'attn' or 'raw'")
        if ts_stats not in ("released", "own"):
            raise PlanError("ts_stats must be 'released' or 'own'")
        self.ts_stats = ts_stats
        self.method, self.until, self.blocks = method, int(until_step), frozenset(blocks)
        self.chunk_size, self.edit_fn, self.ts_logits = int(chunk_size), edit_fn, ts_logits
        self.plan, self.layout = None, None
        self.calls, self.step, self.active_calls = 0, -1, 0

    def begin_video(self, plan):
        self.plan, self.layout = plan, None
        self.calls, self.step, self.active_calls = 0, -1, 0

    def end_video(self):
        self.plan, self.layout = None, None

    def on_forward(self, kwargs):
        if self.plan is None:
            return
        if self.layout is None:
            missing = [n for n in ("position_ids", "video_indices", "text_indices") if n not in kwargs]
            if missing:
                raise PlanError("the H3 transformer was called without %s" % missing)
            layout = Layout(kwargs["position_ids"], kwargs["video_indices"], kwargs["text_indices"])
            if layout.n_text != self.plan["n_tokens"]:
                raise PlanError("H3 encoded %d text rows but the prompt has %d tokens"
                                % (layout.n_text, self.plan["n_tokens"]))
            self.layout = layout
        self.step = self.calls
        self.calls += 1

    def is_active(self, block_index):
        return (self.plan is not None and self.layout is not None and 0 <= self.step < self.until
                and block_index in self.blocks)


class H3EditAttnProcessor:
    """Drop-in for MiniMaxH3AttnProcessor: the original outside the schedule, the edit inside it."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, state, block_index, original, rope_fn=None):
        self.state, self.block_index, self.original = state, block_index, original
        self.rope_fn = rope_fn

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        st = self.state
        if not st.is_active(self.block_index):
            return self.original(attn, hidden_states, rotary_emb, attention_mask)
        if attention_mask is not None:
            raise ValueError("the H3 edit expects the native unpadded packed sequence")
        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query, key, value = (t.unflatten(-1, (attn.heads, -1)) for t in (query, key, value))
        raw = (query, key) if (st.method == TS_ATTN and st.ts_logits == "raw") else None
        query, key = attn.norm_q(query), attn.norm_k(key)
        if rotary_emb is not None:
            rope = self.rope_fn or diffusers_rope()
            query, key = rope(query, *rotary_emb), rope(key, *rotary_emb)
        layout = st.layout.to(query.device)
        plan = st.plan
        if st.method == SWITCHCRAFT:
            bias = switchcraft_bias(query, key, layout, st.edit_fn, plan["event_cols"], plan["shares"])
        else:
            bias = ts_attn_bias(query, key, layout, plan["event_cols"], plan["subject_cols"], plan["shares"],
                                st.chunk_size, raw=raw, stats=st.ts_stats)
        hidden = biased_attention(query, key, value, layout, bias, st.chunk_size)
        st.active_calls += 1
        hidden = hidden.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden))


def install(transformer, state, rope_fn=None):
    """Wrap every transformer block's attention and count the forwards (one per denoising step: H3 is
    guidance-distilled). Returns uninstall()."""
    blocks = list(transformer.transformer_blocks)
    originals = [b.attn.processor for b in blocks]
    for i, (block, original) in enumerate(zip(blocks, originals)):
        block.attn.set_processor(H3EditAttnProcessor(state, i, original, rope_fn))

    def pre_hook(module, args, kwargs):
        state.on_forward(kwargs)

    handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def uninstall():
        handle.remove()
        for block, original in zip(blocks, originals):
            block.attn.set_processor(original)
    return uninstall
