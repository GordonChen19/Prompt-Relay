"""Metrics that are not part of VBench or VBench++ (each has an --no_... flag in evaluate.py).

  clip_t        CLIP text-frame cosine similarity, mean over a segment's core frames, with its local prompt
                (MinT computes CLIP score on clips cut at each event; SwitchCraft reports CLIP-T).
  prompt_match  each segment scored with CLIP-T against every local prompt: the share of segments whose
                own prompt scores highest (a cheap check of event order).
  clip_f        CLIP image cosine similarity of adjacent frames (SwitchCraft): mean, minimum, and the
                minimum inside each switch window.
CLIP is OpenAI's ViT-L/14 through the 'clip' package VBench already installs.
"""
import importlib.util
import os

import numpy as np

CLIP_CONTEXT = 77     # CLIP reads at most 77 tokens of a prompt; the rest is cut off
VICLIP_CONTEXT = 32   # VBench's ViCLIP (overall_consistency) reads at most 32 tokens



# ---------------------------------------------------------------- token counts (no torch needed)
_TOKENIZER = None


def _clip_tokenizer():
    """CLIP's BPE tokenizer, loaded from the 'clip' package files without importing torch."""
    global _TOKENIZER
    if _TOKENIZER is None:
        spec = importlib.util.find_spec("clip")
        if spec is None or not spec.submodule_search_locations:
            return None
        path = os.path.join(list(spec.submodule_search_locations)[0], "simple_tokenizer.py")
        tok_spec = importlib.util.spec_from_file_location("_vbe_clip_tokenizer", path)
        module = importlib.util.module_from_spec(tok_spec)
        tok_spec.loader.exec_module(module)
        _TOKENIZER = module.SimpleTokenizer()
    return _TOKENIZER


def count_tokens(text):
    """Tokens CLIP-style text encoders see (including start and end tokens); None if 'clip' is missing."""
    try:
        tok = _clip_tokenizer()
    except Exception:  # noqa: BLE001 - token counts are only a warning
        return None
    return None if tok is None else len(tok.encode(text)) + 2


def truncation_warnings(texts):
    """Which prompts CLIP (77 tokens) or ViCLIP (32 tokens) will cut short."""
    notes = []
    for label, text in texts:
        n = count_tokens(text)
        if n is None:
            return ["token counts unavailable ('clip' package not installed)"]
        if n > CLIP_CONTEXT:
            notes.append(f"{label}: {n} tokens; CLIP-T reads the first {CLIP_CONTEXT}, "
                         f"overall_consistency the first {VICLIP_CONTEXT}")
        elif n > VICLIP_CONTEXT:
            notes.append(f"{label}: {n} tokens; overall_consistency (ViCLIP) reads the first {VICLIP_CONTEXT}")
    return notes


# ---------------------------------------------------------------- pure scoring (numpy, testable)
def normalise(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def clip_t_scores(frame_emb, text_emb, segments):
    """frame_emb [T, D], text_emb [P, D] (one row per local prompt), segments with 'core' ranges.
    Returns (matrix S[k][j] = mean cosine of segment k's core frames with prompt j, own-prompt scores)."""
    f, t = normalise(frame_emb), normalise(text_emb)
    sims = f @ t.T
    matrix = np.array([[float(sims[s["core"][0]:s["core"][1], j].mean()) for j in range(t.shape[0])]
                       for s in segments[:t.shape[0]]])
    own = [float(matrix[k, k]) for k in range(matrix.shape[0])]
    return matrix, own


def prompt_match_accuracy(matrix):
    """Share of segments whose own prompt has the highest CLIP-T (needs at least 2 prompts)."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] < 2:
        return float("nan")
    return float(np.mean([int(np.argmax(m[k]) == k) for k in range(m.shape[0])]))


def adjacent_similarity(frame_emb):
    """Cosine similarity of frame t and t+1, for t = 0 .. T-2."""
    f = normalise(frame_emb)
    return np.sum(f[:-1] * f[1:], axis=1)


def clip_f_scores(frame_emb, switches):
    """Mean and minimum adjacent-frame similarity, and the minimum over pairs inside switch windows."""
    adj = adjacent_similarity(frame_emb)
    out = {"clip_f_mean": float(adj.mean()), "clip_f_min": float(adj.min())}
    inside = [adj[t] for w in switches for t in range(w["window"][0], w["window"][1] - 1)]
    if switches:
        out["clip_f_switch_min"] = float(min(inside)) if inside else float("nan")
    return out


# ---------------------------------------------------------------- models (need torch)
class ClipScorer:
    def __init__(self, name="ViT-L/14", device="cuda"):
        import clip
        import torch
        self.torch, self.clip, self.device = torch, clip, device
        self.model, self.preprocess = clip.load(name, device=device)
        self.model.eval()

    def frames(self, frames, batch=32):
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for i in range(0, len(frames), batch):
                x = self.torch.stack([self.preprocess(Image.fromarray(f)) for f in frames[i:i + batch]])
                out.append(self.model.encode_image(x.to(self.device)).float().cpu().numpy())
        return np.concatenate(out)

    def texts(self, texts):
        with self.torch.no_grad():
            tokens = self.clip.tokenize(texts, truncate=True).to(self.device)
            return self.model.encode_text(tokens).float().cpu().numpy()
