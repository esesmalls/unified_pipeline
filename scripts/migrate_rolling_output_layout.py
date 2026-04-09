#!/usr/bin/env python3
"""
Migrate rolling inference outputs from legacy layout to init-first layout.

Legacy:
  {output_root}/{DisplayName}/ERA5_6H/{var}[ _surface]_{{init_tag}}.npy
  {output_root}/plots/{registry_slug}/{init_tag}/*.png
  {output_root}/eval_*_{init_tag}/

New:
  {output_root}/{init_tag}/{output_slug}/...
  {output_root}/{init_tag}/plots/{output_slug}/...
  {output_root}/{init_tag}/eval_*/

Use --dry-run first. Pass --output-root multiple times for several result trees.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zk_io.rolling_paths import (
    meta_path,
    npy_dir,
    output_slug_for_display,
    output_slug_for_registry_slug,
    parse_eval_dir_legacy_name,
    parse_init_tag_from_npy_name,
    plot_dir_for_registry_slug,
)

_INIT_TAG_DIR = re.compile(r"^\d{8}T\d{2}$")


def _is_reserved_top(name: str) -> bool:
    return name.startswith("_") or name == "plots"


def _move(src: Path, dest: Path, dry_run: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing: {dest}")
    print(f"  MOVE {src} -> {dest}")
    if not dry_run:
        shutil.move(str(src), str(dest))


def _maybe_rename_nc_slug(
    output_root: Path,
    init_tag: str,
    dry_run: bool,
) -> None:
    nc_root = output_root / init_tag / "nc"
    if not nc_root.is_dir():
        return
    for child in list(nc_root.iterdir()):
        if not child.is_dir():
            continue
        slug = child.name
        want = output_slug_for_registry_slug(slug)
        if want == slug:
            continue
        dest = nc_root / want
        if dest.exists():
            continue
        print(f"  RENAME_NC_DIR {child} -> {dest}")
        if not dry_run:
            child.rename(dest)


def migrate_output_root(output_root: Path, dry_run: bool) -> None:
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        print(f"[skip] not a directory: {output_root}")
        return
    print(f"=== {output_root} ===")

    # 1) Legacy model folders with ERA5_6H
    for model_dir in sorted(output_root.iterdir()):
        if not model_dir.is_dir() or _is_reserved_top(model_dir.name):
            continue
        era = model_dir / "ERA5_6H"
        if not era.is_dir():
            continue
        display_name = model_dir.name
        out_slug = output_slug_for_display(display_name)

        for f in sorted(era.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() != ".npy":
                continue
            parsed = parse_init_tag_from_npy_name(f.name)
            if not parsed:
                print(f"  SKIP unparsed npy name: {f}")
                continue
            _var_stem, init_tag = parsed
            dest_dir = npy_dir(output_root, init_tag, display_name)
            dest = dest_dir / f.name
            try:
                _move(f, dest, dry_run)
            except FileExistsError as e:
                print(f"  WARN {e}")

        for f in sorted(era.iterdir()):
            if f.is_file() and f.name.startswith("meta_") and f.suffix == ".json":
                # meta_{init_tag}.json
                rest = f.name[5:-5]
                if len(rest) >= 10 and "T" in rest:
                    init_tag = rest
                    dest = meta_path(output_root, init_tag, display_name)
                    try:
                        _move(f, dest, dry_run)
                    except FileExistsError as e:
                        print(f"  WARN {e}")

        # Remove empty ERA5_6H / model dir
        if not dry_run:
            try:
                if era.exists() and not any(era.iterdir()):
                    era.rmdir()
                    if model_dir.exists() and not any(model_dir.iterdir()):
                        model_dir.rmdir()
            except OSError:
                pass

    # 2) plots: plots/{registry_slug}/{init_tag}/* -> {init_tag}/plots/{output_slug}/
    plots_root = output_root / "plots"
    if plots_root.is_dir():
        for slug_dir in sorted(plots_root.iterdir()):
            if not slug_dir.is_dir():
                continue
            reg_slug = slug_dir.name
            for itag_subdir in sorted(slug_dir.iterdir()):
                if not itag_subdir.is_dir():
                    continue
                init_tag = itag_subdir.name
                dest_base = plot_dir_for_registry_slug(output_root, init_tag, reg_slug)
                for f in sorted(itag_subdir.iterdir()):
                    if not f.is_file():
                        continue
                    dest = dest_base / f.name
                    try:
                        _move(f, dest, dry_run)
                    except FileExistsError as e:
                        print(f"  WARN {e}")
                if not dry_run:
                    try:
                        if itag_subdir.exists() and not any(itag_subdir.iterdir()):
                            itag_subdir.rmdir()
                    except OSError:
                        pass
            if not dry_run:
                try:
                    if slug_dir.exists() and not any(slug_dir.iterdir()):
                        slug_dir.rmdir()
                except OSError:
                    pass
        if not dry_run:
            try:
                if plots_root.exists() and not any(plots_root.iterdir()):
                    plots_root.rmdir()
            except OSError:
                pass

    # 3) eval_*_{init_tag} at output_root root
    for p in sorted(output_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("eval_"):
            continue
        parsed = parse_eval_dir_legacy_name(p.name)
        if not parsed:
            continue
        new_base, init_tag = parsed
        dest = output_root / init_tag / new_base
        if dest.exists():
            print(f"  SKIP eval dest exists: {dest}")
            continue
        print(f"  MOVE_DIR {p} -> {dest}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            p.rename(dest)

    # 4) Normalize NC subdir names under {init_tag}/nc/
    for p in sorted(output_root.iterdir()):
        if not p.is_dir() or _is_reserved_top(p.name):
            continue
        if _INIT_TAG_DIR.match(p.name):
            _maybe_rename_nc_slug(output_root, p.name, dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        action="append",
        dest="output_roots",
        required=True,
        help="Result tree to migrate (repeat for multiple roots)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves only",
    )
    args = ap.parse_args()
    for root in args.output_roots:
        migrate_output_root(root, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
