"""Read and check the prompt-suite JSON (one entry per video).

An entry:
    {
      "id": "p001",                         # optional, defaults to the file name without .mp4
      "file_name": "p001.mp4",              # required
      "file_dir": "outputs/wan22_relay",    # optional when --videos_dir is given (it overrides this)
      "global_prompt": "...",               # what stays true for the whole video
      "local_prompts": ["...", "..."],      # the events, in order; 0 or 1 = the video is not cut
      "images": ["inputs/img1.png", "inputs/img2.png"],  # optional (Visual Relay): image k goes with segment k
      "image_latents": [0, 14],             # optional (Visual Relay): the latent each image is placed at
      "switch_fractions": [0.688]           # optional, rarely needed: where the prompt switches (segment 2, 3, ...
                                            # start), as fractions of the video length; default: even split
    }

The file may hold one entry (an object), a list of entries, or {"prompts": [entries]}.
Any other field is kept and copied into the results ("extra"). Relative paths are relative to the folder
you run the command from.
"""
import json
import os

KNOWN_FIELDS = {"id", "file_name", "file_dir", "global_prompt", "local_prompts", "images", "image_latents",
                "switch_fractions"}
RENAMED_FIELDS = {"cuts": "switch_fractions"}
VIDEO_EXTENSIONS = (".mp4",)


class SuiteError(ValueError):
    """The prompt JSON does not follow the format in README.md."""


def _check_entry(entry, position, videos_dir):
    where = f"entry {position}"
    if not isinstance(entry, dict):
        raise SuiteError(f"{where}: must be a JSON object, got {type(entry).__name__}")
    file_name = entry.get("file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        raise SuiteError(f"{where}: 'file_name' is required (for example \"p001.mp4\")")
    if not file_name.lower().endswith(VIDEO_EXTENSIONS):
        raise SuiteError(f"{where}: 'file_name' must end in .mp4, got {file_name!r}")
    if os.path.basename(file_name) != file_name:
        raise SuiteError(f"{where}: 'file_name' must be a bare file name; put folders in 'file_dir'")
    where = f"entry {position} ({file_name})"

    file_dir = entry.get("file_dir")
    if file_dir is not None and not isinstance(file_dir, str):
        raise SuiteError(f"{where}: 'file_dir' must be a string")
    folder = videos_dir if videos_dir is not None else file_dir
    if not folder:
        raise SuiteError(f"{where}: no 'file_dir' and no --videos_dir, so the video cannot be found")

    global_prompt = entry.get("global_prompt", "")
    if not isinstance(global_prompt, str):
        raise SuiteError(f"{where}: 'global_prompt' must be a string")
    global_prompt = global_prompt.strip()

    local_prompts = entry.get("local_prompts", [])
    if not isinstance(local_prompts, list) or not all(isinstance(p, str) and p.strip() for p in local_prompts):
        raise SuiteError(f"{where}: 'local_prompts' must be a list of non-empty strings")
    local_prompts = [p.strip() for p in local_prompts]
    if not global_prompt and not local_prompts:
        raise SuiteError(f"{where}: needs a 'global_prompt' or at least one 'local_prompts' entry")
    n_segments = max(1, len(local_prompts))

    images = entry.get("images", [])
    if images is None:
        images = []
    if not isinstance(images, list) or not all(i is None or (isinstance(i, str) and i.strip()) for i in images):
        raise SuiteError(f"{where}: 'images' must be a list of paths (or null)")
    if len(images) > n_segments:
        raise SuiteError(f"{where}: {len(images)} images but only {n_segments} segment(s); "
                         "image k is compared with segment k")
    images = list(images) + [None] * (n_segments - len(images))

    for old, new in RENAMED_FIELDS.items():
        if old in entry:
            raise SuiteError(f"{where}: '{old}' is now called '{new}'")
    switch = entry.get("switch_fractions")
    if switch is not None:
        if n_segments == 1:
            raise SuiteError(f"{where}: 'switch_fractions' given but the video has a single segment")
        if (not isinstance(switch, list) or len(switch) != n_segments - 1
                or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in switch)):
            raise SuiteError(f"{where}: 'switch_fractions' must be a list of {n_segments - 1} numbers between 0 and 1"
                             " (one per prompt switch)")
        switch = [float(c) for c in switch]
        if not all(0.0 < c < 1.0 for c in switch) or any(b <= a for a, b in zip(switch, switch[1:])):
            raise SuiteError(f"{where}: 'switch_fractions' must be increasing and strictly between 0 and 1, "
                             f"got {switch}")

    latents = entry.get("image_latents")
    if latents is not None:
        given = [i for i in images if i is not None]
        if (not isinstance(latents, list) or len(latents) != len(given)
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in latents)):
            raise SuiteError(f"{where}: 'image_latents' must be a list of {len(given)} whole numbers, "
                             "one per image")
        if latents and (latents[0] != 0 or any(b <= a for a, b in zip(latents, latents[1:]))):
            raise SuiteError(f"{where}: 'image_latents' must start at 0 (the first frame) and increase, "
                             f"got {latents}")

    entry_id = entry.get("id", os.path.splitext(file_name)[0])
    if not isinstance(entry_id, (str, int)) or isinstance(entry_id, bool) or str(entry_id).strip() == "":
        raise SuiteError(f"{where}: 'id' must be a non-empty string")

    return {
        "id": str(entry_id).strip(),
        "file_name": file_name,
        "video_path": os.path.join(folder, file_name),
        "global_prompt": global_prompt,
        "local_prompts": local_prompts,
        "n_segments": n_segments,
        "images": images,
        "switch_fractions": switch,
        "image_latents": latents,
        "extra": {k: v for k, v in entry.items() if k not in KNOWN_FIELDS},
    }


def load_suite(path, videos_dir=None):
    """Return the checked entries of one prompt JSON. --videos_dir (videos_dir) overrides every 'file_dir'."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise SuiteError(f"{path} is not valid JSON: {err}") from err
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        entries = data["prompts"]
    elif isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        raise SuiteError(f"{path}: expected an object or a list of objects")
    if not entries:
        raise SuiteError(f"{path}: no entries")

    checked, errors = [], []
    for position, entry in enumerate(entries):
        try:
            checked.append(_check_entry(entry, position, videos_dir))
        except SuiteError as err:
            errors.append(str(err))
    seen = {}
    for e in checked:
        for key, label in ((e["id"], "id"), (e["file_name"], "file_name")):
            if (label, key) in seen:
                errors.append(f"duplicate {label} {key!r}")
            seen[(label, key)] = True
    if errors:
        raise SuiteError("problems in " + path + ":\n  " + "\n  ".join(errors))
    return checked


def whole_video_text(entry):
    """The text the whole video is compared with: the global prompt, or all local prompts joined."""
    return entry["global_prompt"] or " ".join(entry["local_prompts"])


def segment_texts(entry):
    """The text each segment is compared with (empty when the entry has no local prompts)."""
    return list(entry["local_prompts"])
