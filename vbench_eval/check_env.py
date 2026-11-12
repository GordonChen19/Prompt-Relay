"""Check the VBench environment before a real run: versions, VBench's modules for every dimension we use,
the download tools VBench calls, and the GPU. Loads no model weights.

    python vbench_eval/check_env.py [--vbench_root PATH]      (default: $VBENCH_ROOT, else third_party/VBench)
"""
import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vbeval import metrics_vbench as MV  # noqa: E402

VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
RESULTS = []


def report(status, what, detail=""):
    RESULTS.append(status)
    print(f"[{status}] {what}" + (f": {detail}" if detail else ""))


def version(module):
    try:
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001 - report every failure, keep checking
        return exc


def check_version(module, label, ok=None, hint=""):
    v = version(module)
    if isinstance(v, Exception):
        report("FAIL", label, f"cannot import {module} ({type(v).__name__}: {v}) {hint}".strip())
    elif ok is not None and not ok(v):
        report("FAIL", label, f"{v} {hint}".strip())
    else:
        report("OK", label, str(v))
    return v


def vtuple(v):
    out = []
    for part in str(v).split("+")[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vbench_root", default=MV.default_vbench_root())
    a = ap.parse_args()

    py = sys.version_info
    report("OK" if (py.major, py.minor) == (3, 10) else "WARN", "python",
           f"{platform.python_version()} ({sys.executable})" + ("" if (py.major, py.minor) == (3, 10)
                                                               else "; VBench was built for 3.10"))

    torch = version("torch")
    if isinstance(torch, Exception):
        report("FAIL", "torch", f"cannot import ({torch})")
    else:
        import torch as T
        cuda = T.version.cuda
        report("OK" if cuda in ("11.6", "11.7", "11.8", "12.1") else "WARN", "torch",
               f"{T.__version__}, built for CUDA {cuda}" + ("" if cuda in ("11.6", "11.7", "11.8", "12.1")
                                                            else "; VBench asks for CUDA 11.6-11.8 or 12.1"))
        if T.cuda.is_available():
            report("OK", "GPU", f"{T.cuda.device_count()} visible, {T.cuda.get_device_name(0)}")
        else:
            report("WARN", "GPU", "none visible (fine for --dry_run and tests, needed for a real run)")

    check_version("numpy", "numpy < 2", lambda v: vtuple(v) < (2,), "(VBench needs numpy < 2)")
    check_version("transformers", "transformers == 4.33.2", lambda v: str(v) == "4.33.2",
                  "(VBench pins 4.33.2; do not share this environment with Wan or MiniMax)")
    check_version("timm", "timm <= 1.0.12", lambda v: vtuple(v) <= (1, 0, 12))
    check_version("pyiqa", "pyiqa (imaging_quality)")
    check_version("clip", "openai clip (CLIP-T, CLIP-F, VBench)")
    check_version("decord", "decord (VBench video reading)")
    check_version("cv2", "opencv")
    check_version("dreamsim", "dreamsim (i2v_background)", hint="(or run with --skip i2v_background)")
    check_version("peft", "peft (DreamSim)")
    try:
        import imageio_ffmpeg
        report("OK", "ffmpeg for writing clips", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "ffmpeg for writing clips", f"{exc} (pip install imageio-ffmpeg)")

    for tool in ("wget", "unzip", "git"):
        path = shutil.which(tool)
        report("OK" if path else "FAIL", f"{tool} on PATH", path or "missing (VBench downloads its models with wget/unzip)")

    root = a.vbench_root
    if not root or not os.path.isdir(os.path.join(root, "vbench")):
        report("FAIL", "VBench clone", f"{root!r} has no 'vbench' folder: run "
               "'git submodule update --init vbench_eval/third_party/VBench' or pass --vbench_root")
    else:
        try:
            commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            commit = "unknown"
        report("OK" if commit == VBENCH_COMMIT else "WARN", "VBench commit",
               f"{commit[:7]} at {root}" + ("" if commit == VBENCH_COMMIT else
                                            f"; this tool was checked against {VBENCH_COMMIT[:7]}"))
        try:
            MV.import_vbench(root)
            report("OK", "import vbench")
        except Exception as exc:  # noqa: BLE001
            report("FAIL", "import vbench", f"{type(exc).__name__}: {exc}")
        for dim in MV.VBENCH_DIMS + MV.I2V_DIMS:
            pkg = "vbench" if dim in MV.VBENCH_DIMS else "vbench2_beta_i2v"
            try:
                importlib.import_module(f"{pkg}.{dim}")
                report("OK", f"VBench module {pkg}.{dim}")
            except Exception as exc:  # noqa: BLE001
                report("FAIL", f"VBench module {pkg}.{dim}", f"{type(exc).__name__}: {exc}")
        try:
            importlib.import_module("pyiqa.archs.musiq_arch")
            report("OK", "pyiqa MUSIQ model code")
        except Exception as exc:  # noqa: BLE001
            report("FAIL", "pyiqa MUSIQ model code", f"{type(exc).__name__}: {exc}")

    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    lines = (pip.stdout + pip.stderr).strip().splitlines()
    report("OK" if pip.returncode == 0 else "WARN", "pip check",
           lines[0] if pip.returncode == 0 and lines else "; ".join(lines[:6]))

    fails, warns = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n{len(RESULTS) - fails - warns} OK, {warns} warning(s), {fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
