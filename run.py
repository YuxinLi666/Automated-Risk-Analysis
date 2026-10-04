"""Convenient repository-local entry points. No training is run on import."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {
        "test": ["-m", "unittest", "discover", "-s", "tests", "-v"],
        "demo": ["scripts/demo_pipeline.py"],
        "cli": ["-m", "engine_fault_nlp"],
        "train": ["scripts/run_macbert_mgp_module_ablation.py", "--variants", "full", "--seeds", "42", "--save-checkpoints"],
        "ablate": ["scripts/run_macbert_mgp_module_ablation.py"],
        "predict": ["scripts/predict_macbert_mgp.py"],
        "rank": ["scripts/rank_fault_paths.py"],
        "audit": ["tools/verify_repository.py"],
    }
    parser.add_argument("command", choices=["doctor", *commands])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        packages = {}
        for name in ("numpy", "pandas", "torch", "transformers", "safetensors"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not installed"
        print(json.dumps({"python": platform.python_version(), "platform": platform.system(),
                          "packages": packages, "note": "Package versions do not prove GPU availability."}, indent=2))
        return 0 if sys.version_info >= (3, 10) else 1
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call([sys.executable, *commands[args.command], *args.args], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
