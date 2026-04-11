#!/usr/bin/env python3
"""
Compare stepwise (in-process JAX) NPY outputs against the cache-based
official JAX baseline.

Usage (after both models have completed rolling inference for the same init_tag):

  python scripts/compare_stepwise_vs_cache.py \
      --init-tag 20260303T12 \
      --baseline-dir /public/share/aciwgvx1jd/gc_oper_rollout_cache/GraphCast_official/ERA5_6H \
      --stepwise-dir /public/share/aciwgvx1jd/LYQ/gundong/GunDong_Infer_result_12h_stepwise/GC_Stepwise/ERA5_6H

Prints per-variable / per-step max-abs-diff, RMSE, and overall statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VARS = ("u10", "v10", "t2m", "msl")


def _load(root: Path, var: str, tag: str) -> np.ndarray:
    path = root / f"{var}_{tag}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(str(path)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-tag", required=True, help="e.g. 20260303T12")
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Directory with cache-based NPY files")
    ap.add_argument("--stepwise-dir", type=Path, required=True,
                    help="Directory with stepwise NPY files")
    ap.add_argument("--rtol", type=float, default=1e-4,
                    help="Relative tolerance for 'close' check (default 1e-4)")
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="Absolute tolerance for 'close' check (default 1e-3)")
    args = ap.parse_args()

    tag = args.init_tag
    all_close = True

    for var in VARS:
        try:
            baseline = _load(args.baseline_dir, var, tag)
            stepwise = _load(args.stepwise_dir, var, tag)
        except FileNotFoundError as e:
            print(f"[{var}] SKIP — file not found: {e}")
            all_close = False
            continue

        if baseline.shape != stepwise.shape:
            print(f"[{var}] SHAPE MISMATCH  baseline={baseline.shape}  "
                  f"stepwise={stepwise.shape}")
            all_close = False
            continue

        diff = np.abs(baseline - stepwise)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        rmse = float(np.sqrt(np.mean((baseline - stepwise) ** 2)))
        close = bool(np.allclose(baseline, stepwise, rtol=args.rtol, atol=args.atol))

        n_steps = baseline.shape[0]
        print(f"[{var}]  steps={n_steps}  max_abs_diff={max_abs:.6e}  "
              f"mean_abs_diff={mean_abs:.6e}  rmse={rmse:.6e}  "
              f"allclose(rtol={args.rtol},atol={args.atol})={close}")

        if not close:
            all_close = False
            worst_step = int(np.argmax(diff.reshape(n_steps, -1).max(axis=1)))
            step_max = float(diff[worst_step].max())
            print(f"        worst step={worst_step}  (max_abs={step_max:.6e})")

    print()
    if all_close:
        print("RESULT: ALL VARIABLES MATCH within tolerance.")
    else:
        print("RESULT: MISMATCH detected — see details above.")


if __name__ == "__main__":
    main()
