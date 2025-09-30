#!/usr/bin/env python3
"""
Sanity runner:
- By default does a fast compile-only check (py_compile) to catch syntax/import errors.
- Optionally run full execution per-file by setting exec_mode=True below.
- Sets PYTHONPATH so local packages (src, models) are importable.
"""
import subprocess
import sys
import os
from pathlib import Path

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUDE_PARTS = {
    "__pycache__", ".venv", "venv", "env", ".git",
    "sanity.py", "notebooks", "artifacts", "models/checkpoints",
}
# Files that we *never* want to execute during a full run (training, streamlit, CLI)
SKIP_FULL_EXEC = {
    "src/train_model.py",
    "models/trainer.py",
    "app/streamlit_app.py",
    "gan/ctgan_wrapper.py",  # optional - may require heavy deps
}

# If True -> actually execute each file (may run training). If False -> use py_compile.
exec_mode = True
# ----------------------------

def should_skip(path: Path):
    # skip files in excluded paths or this script itself
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    # skip files explicitly in SKIP_FULL_EXEC when exec_mode True
    rel = path.relative_to(PROJECT_ROOT)
    if exec_mode and str(rel) in SKIP_FULL_EXEC:
        return True
    # don't run non-top-level scripts (e.g., __init__.py can be skipped if desired)
    return False

def run_python_files(base_dir: Path):
    py_files = sorted([p for p in base_dir.rglob("*.py") if not should_skip(p)])
    print(f"\n🔍 Found {len(py_files)} Python files to check.\n")

    env = os.environ.copy()
    # ensure local imports work
    env["PYTHONPATH"] = str(base_dir) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    for file in py_files:
        print(f"➡️ {('Executing' if exec_mode else 'Compiling')} {file} ...")
        cmd = [sys.executable]
        if exec_mode:
            cmd.append(str(file))
        else:
            # fast syntax + import check
            cmd.extend(["-m", "py_compile", str(file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, cwd=str(PROJECT_ROOT))
            if result.stdout:
                print(f"   ✅ Output:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error in {file}")
            if e.stdout:
                print(f"   ---- STDOUT ----\n{e.stdout.strip()}")
            if e.stderr:
                print(f"   ---- STDERR ----\n{e.stderr.strip()}")
        print("-" * 60)

if __name__ == "__main__":
    run_python_files(PROJECT_ROOT)

