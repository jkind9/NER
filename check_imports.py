#!/usr/bin/env python
"""
check_environment.py

Reads module names from a requirements.txt file, verifies each can be imported,
and checks that CUDA is available for PyTorch.
"""

import argparse
import importlib
import re
import sys

def read_modules(req_path: str) -> list[str]:
    """
    Parse the given requirements.txt and extract the package names.
    Lines starting with '#' or blank lines are skipped.
    Version specifiers (<, <=, ==, >=, >, ~=) are removed.
    """
    modules = []
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # split on the first occurrence of any version specifier
            name = re.split(r"[<>=~!]", line, maxsplit=1)[0].strip()
            if name:
                modules.append(name)
    return modules

def check_imports(modules: list[str]) -> bool:
    ok = True
    print("Checking imports…")
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"[ OK ] {mod}")
        except ImportError as e:
            print(f"[FAIL] {mod}: {e}")
            ok = False
    return ok

def check_cuda() -> bool:
    print("\nChecking CUDA availability…")
    try:
        import torch
    except ImportError:
        print("[FAIL] torch not installed; cannot check CUDA")
        return False

    available = torch.cuda.is_available()
    print(f"[ INFO ] torch.cuda.is_available() = {available}")
    if available:
        print(f"[ INFO ] CUDA device count: {torch.cuda.device_count()}")
    return available

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements",
        type=str,
        default="requirements.txt",
        help="Path to requirements.txt",
    )
    args = parser.parse_args()

    try:
        modules = read_modules(args.requirements)
    except FileNotFoundError:
        print(f"[FAIL] requirements file not found: {args.requirements}", file=sys.stderr)
        sys.exit(1)

    imports_ok = check_imports(modules)
    cuda_ok    = check_cuda()

    print()
    if not imports_ok:
        print("Environment check FAILED: missing modules.", file=sys.stderr)
        sys.exit(1)
    if not cuda_ok:
        print("Environment check WARNING: CUDA not available.", file=sys.stderr)
        sys.exit(2)
    print("Environment check PASSED 🎉")

if __name__ == "__main__":
    main()
