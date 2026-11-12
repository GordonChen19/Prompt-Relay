"""Reading videos, writing the short clips VBench scores, and the per-video sidecar."""
import json
import os

import numpy as np


def read_video(path):
    """All frames as uint8 RGB [T, H, W, 3] and the frame rate."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[..., ::-1])
    cap.release()
    if not frames:
        raise IOError(f"no frames could be read from {path}")
    if fps <= 0:
        raise IOError(f"{path} reports no frame rate")
    return np.ascontiguousarray(np.stack(frames)), float(fps)


def write_clip(frames, fps, path):
    """Save frames [T, H, W, 3] (RGB uint8) as a near-lossless H.264 mp4 at the original frame rate."""
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
                                ffmpeg_log_level="error",
                                ffmpeg_params=["-crf", "1", "-pix_fmt", "yuv420p"])
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()
    return path


def save_rgb_image(src, dst):
    """Copy an input image as 3-channel RGB PNG (VBench-I2V cannot read images with an alpha channel)."""
    from PIL import Image
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst)
    return dst


def sidecar_path(video_path):
    """'<folder>/<name>.meta.json' next to '<folder>/<name>.mp4'."""
    return os.path.splitext(video_path)[0] + ".meta.json"


def read_sidecar(video_path):
    """The generation record written next to a video ({} when there is none)."""
    path = sidecar_path(video_path)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def write_sidecar(video_path, data):
    """Merge data into the video's sidecar."""
    record = read_sidecar(video_path)
    record.update(data)
    with open(sidecar_path(video_path), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return sidecar_path(video_path)
