#!/usr/bin/env python3
"""
Lift contents of ``ECMWF_Init_Infer_result_*`` directories into a single
``ecmwf_init`` root so layout is::

  {ecmwf_base}/{init_tag}/pangu/ ...
  {ecmwf_base}/{init_tag}/fengwu/ ...
  {ecmwf_base}/_capability_memory/
  {ecmwf_base}/_gc_oper_cache/

When the same ``init_tag`` already exists and a full merge would overwrite
files (e.g. AB test trees), the source init directory is moved to
``{init_tag}_{suffix}`` where *suffix* is derived from the source folder name.

Run from anywhere; repo root is added to ``sys.path`` only if needed for imports
(none required here).

Example::

  python scripts/flatten_ecmwf_init_roots.py \\
    --ecmwf-base /public/share/aciwgvx1jd/LYQ/ecmwf_init --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

_INIT_TAG = re.compile(r"^\d{8}T\d{2}$")


def _source_suffix(source_root: Path) -> str:
    name = source_root.name
    if name.startswith("ECMWF_Init_Infer_result_"):
        return name[len("ECMWF_Init_Infer_result_") :]
    return name


def _list_result_roots(ecmwf_base: Path) -> List[Path]:
    roots = [
        p
        for p in ecmwf_base.iterdir()
        if p.is_dir() and p.name.startswith("ECMWF_Init_Infer_result_")
    ]

    def sort_key(p: Path) -> Tuple[int, Tuple]:
        n = p.name
        m = re.match(r"^ECMWF_Init_Infer_result_(\d+)h$", n)
        if m:
            return (0, (int(m.group(1)),))
        return (1, (n,))

    return sorted(roots, key=sort_key)


def _merge_capability_json(dst: Path, src: Path, dry_run: bool) -> None:
    if not src.is_file():
        return
    if not dst.is_file():
        print(f"  MOVE {src} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return
    print(f"  MERGE_JSON {src} into {dst}")
    if dry_run:
        return
    try:
        a = json.loads(dst.read_text(encoding="utf-8"))
        b = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARN skip JSON merge ({e}): keep {dst}")
        return
    for key in ("sources", "models"):
        if key in b and isinstance(b[key], dict):
            slot = a.setdefault(key, {})
            if not isinstance(slot, dict):
                slot = {}
                a[key] = slot
            for k, v in b[key].items():
                slot.setdefault(k, v)
    dst.write_text(json.dumps(a, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        src.unlink()
    except OSError:
        pass


def _merge_tree(src: Path, dst: Path, dry_run: bool) -> None:
    """Merge directory src into existing dst (dirs recurse, files skip if dest exists)."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        target = dst / child.name
        if child.is_dir():
            if target.exists() and target.is_dir():
                _merge_tree(child, target, dry_run)
            elif target.exists():
                print(f"  WARN skip dir {child}: {target} exists and is not a dir")
            else:
                print(f"  MOVE {child} -> {target}")
                if not dry_run:
                    shutil.move(str(child), str(target))
        else:
            if target.exists():
                print(f"  SKIP file (exists): {target}")
            else:
                print(f"  MOVE {child} -> {target}")
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(child), str(target))
    if not dry_run:
        try:
            src.rmdir()
        except OSError:
            pass


def _place_init_dir(
    init_src: Path,
    dest_base: Path,
    source_root: Path,
    dry_run: bool,
    placed_tags: Optional[Set[str]] = None,
) -> None:
    """
    Main hour buckets (0h/6h/12h/18h): merge into existing ``init_tag`` dir when present.
    AB / other ``ECMWF_Init_Infer_result_*`` trees: if ``init_tag`` taken, use
    ``{init_tag}_{suffix}`` so experiments are not overwritten.
    ``placed_tags`` tracks init_tag names already at dest for accurate ``--dry-run``.
    """
    tag = init_src.name
    dest = dest_base / tag
    suffix = _source_suffix(source_root)
    is_main = suffix in ("0h", "6h", "12h", "18h")
    logically_exists = dest.exists() or (placed_tags is not None and tag in placed_tags)

    if not is_main:
        if logically_exists:
            alt = dest_base / f"{tag}_{suffix}"
            print(f"MOVE_DIR {init_src} -> {alt} (side experiment; canonical {tag} exists)")
            if not dry_run:
                if alt.exists():
                    raise FileExistsError(str(alt))
                shutil.move(str(init_src), str(alt))
            return
        print(f"MOVE_DIR {init_src} -> {dest}")
        if not dry_run:
            shutil.move(str(init_src), str(dest))
        if placed_tags is not None:
            placed_tags.add(tag)
        return

    if not logically_exists:
        print(f"MOVE_DIR {init_src} -> {dest}")
        if not dry_run:
            shutil.move(str(init_src), str(dest))
        if placed_tags is not None:
            placed_tags.add(tag)
        return

    print(f"MERGE_DIR {init_src} -> {dest}")
    _merge_tree(init_src, dest, dry_run)
    if placed_tags is not None:
        placed_tags.add(tag)


def _lift_special_dir(name: str, source_root: Path, dest_base: Path, dry_run: bool) -> None:
    src = source_root / name
    if not src.is_dir():
        return
    dst = dest_base / name
    if name == "_capability_memory":
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.name == "source_model_capabilities.json":
                _merge_capability_json(dst / f.name, f, dry_run)
            else:
                target = dst / f.name
                if target.exists():
                    print(f"  SKIP (exists) {target}")
                else:
                    print(f"  MOVE {f} -> {target}")
                    if not dry_run:
                        shutil.move(str(f), str(target))
        if not dry_run and src.exists():
            shutil.rmtree(src, ignore_errors=True)
        return

    # _gc_oper_cache and anything else: merge dirs
    if not dst.exists():
        print(f"MOVE_DIR {src} -> {dst}")
        if not dry_run:
            shutil.move(str(src), str(dst))
        return
    print(f"MERGE_DIR {src} -> {dst}")
    _merge_tree(src, dst, dry_run)
    if not dry_run and src.exists():
        shutil.rmtree(src, ignore_errors=True)


def cleanup_stale_ecmwf_result_roots(ecmwf_base: Path, dry_run: bool) -> None:
    """Remove leftover ``ECMWF_Init_Infer_result_*`` dirs (e.g. after partial lift)."""
    stale = sorted(
        p for p in ecmwf_base.glob("ECMWF_Init_Infer_result_*") if p.is_dir()
    )
    if not stale:
        return
    print(f"=== cleanup {len(stale)} stale ECMWF_Init_Infer_result_* root(s) ===")
    dst_json = ecmwf_base / "_capability_memory/source_model_capabilities.json"
    for p in stale:
        sj = p / "_capability_memory/source_model_capabilities.json"
        if sj.is_file():
            print(f"  merge capability from {p.name}")
            if not dry_run:
                _merge_capability_json(dst_json, sj, dry_run=False)
        print(f"  RMTREE {p}")
        if not dry_run:
            shutil.rmtree(p, ignore_errors=True)


def run_flatten(ecmwf_base: Path, dry_run: bool) -> None:
    ecmwf_base = Path(ecmwf_base).resolve()
    if not ecmwf_base.is_dir():
        raise SystemExit(f"Not a directory: {ecmwf_base}")

    roots = _list_result_roots(ecmwf_base)
    if not roots:
        cleanup_stale_ecmwf_result_roots(ecmwf_base, dry_run)
        return

    placed_tags: Set[str] = set()
    for p in ecmwf_base.iterdir():
        if p.is_dir() and _INIT_TAG.match(p.name):
            placed_tags.add(p.name)

    for source_root in roots:
        print(f"=== lift {source_root.name} ===")
        for child in sorted(source_root.iterdir()):
            if child.name.startswith("."):
                continue
            if _INIT_TAG.match(child.name) and child.is_dir():
                _place_init_dir(
                    child, ecmwf_base, source_root, dry_run, placed_tags,
                )
            elif child.name in ("_capability_memory", "_gc_oper_cache"):
                _lift_special_dir(child.name, source_root, ecmwf_base, dry_run)
            else:
                print(f"  WARN unhandled top-level entry: {child}")

        if not dry_run:
            try:
                leftovers = list(source_root.iterdir())
                if not leftovers:
                    source_root.rmdir()
                    print(f"  REMOVED_EMPTY {source_root}")
                else:
                    print(f"  WARN not removing {source_root}: still has {leftovers}")
            except OSError as e:
                print(f"  WARN could not clean {source_root}: {e}")

    cleanup_stale_ecmwf_result_roots(ecmwf_base, dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ecmwf-base",
        type=Path,
        default=Path("/public/share/aciwgvx1jd/LYQ/ecmwf_init"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_flatten(args.ecmwf_base, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
