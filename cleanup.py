"""
cleanup.py — remove accumulated model downloads, caches, and training artefacts.

What it targets:
  1. HuggingFace model cache   (~/.cache/huggingface/hub/)
  2. HuggingFace dataset cache (~/.cache/huggingface/datasets/)
  3. SFT adapter output        (./poet-sft-lora/)
  4. ChromaDB                  (./chroma_db/)
  5. Python bytecode           (**/__pycache__/)

Run:
    python cleanup.py              # shows sizes, asks before deleting
    python cleanup.py --dry-run    # shows sizes only, deletes nothing
    python cleanup.py --yes        # skips confirmation prompt
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

HF_CACHE       = Path.home() / ".cache" / "huggingface" / "hub"
HF_DATASETS    = Path.home() / ".cache" / "huggingface" / "datasets"
SFT_ADAPTER    = Path("./poet-sft-lora")
CHROMA_DB      = Path("./chroma_db")
PROJECT_ROOT   = Path(__file__).parent


def get_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1e9
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e9


def find_pycache_dirs(root: Path):
    return list(root.rglob("__pycache__"))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(targets: list[tuple[str, Path, float]]) -> None:
    print("\nTargets found:\n")
    total = 0.0
    for label, path, size_gb in targets:
        status = f"{size_gb:.2f} GB" if path.exists() else "not found"
        print(f"  {'[x]' if path.exists() else '[ ]'}  {label:<35} {status}")
        total += size_gb
    print(f"\n  Total reclaimable: {total:.2f} GB\n")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete(path: Path, label: str) -> None:
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  Deleted  {label}")
    except Exception as e:
        print(f"  FAILED   {label}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up model downloads and caches")
    parser.add_argument("--dry-run", action="store_true", help="Show sizes only, delete nothing")
    parser.add_argument("--yes",     action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--keep-chroma", action="store_true",
        help="Keep chroma_db (preserves agent memories across runs)"
    )
    args = parser.parse_args()

    pycache_dirs = find_pycache_dirs(PROJECT_ROOT)

    targets = [
        ("HuggingFace model cache",    HF_CACHE,    get_size_gb(HF_CACHE)),
        ("HuggingFace dataset cache",  HF_DATASETS, get_size_gb(HF_DATASETS)),
        ("SFT adapter (poet-sft-lora)", SFT_ADAPTER, get_size_gb(SFT_ADAPTER)),
    ]

    if not args.keep_chroma:
        targets.append(("ChromaDB (agent memories)", CHROMA_DB, get_size_gb(CHROMA_DB)))

    pycache_size = sum(get_size_gb(p) for p in pycache_dirs)
    if pycache_dirs:
        targets.append((f"__pycache__ ({len(pycache_dirs)} dirs)", Path("__pycache__"), pycache_size))

    report(targets)

    if args.dry_run:
        print("Dry run — nothing deleted.")
        return

    if not args.yes:
        answer = input("Delete all targets above? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    delete(HF_CACHE,    "HuggingFace model cache")
    delete(HF_DATASETS, "HuggingFace dataset cache")
    delete(SFT_ADAPTER, "SFT adapter (poet-sft-lora)")

    if not args.keep_chroma:
        delete(CHROMA_DB, "ChromaDB (agent memories)")

    for p in pycache_dirs:
        delete(p, str(p.relative_to(PROJECT_ROOT)))

    print("\nDone.")


if __name__ == "__main__":
    main()
