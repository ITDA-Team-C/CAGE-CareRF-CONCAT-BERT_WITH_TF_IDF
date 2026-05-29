"""
GNN (FINAL) / LGBM-only / LGBM+GNN Stack 의 5x 결과를 한 표로 집계.

사용:
    python -m src.training.aggregate_final
    python -m src.training.aggregate_final --gnn-dir outputs/benchmark/CHEB
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


METRICS = ("pr_auc", "macro_f1", "roc_auc", "recall_pos", "recall_neg", "g_mean")


def _collect_gnn(gnn_dir: str):
    out = defaultdict(list)
    files = sorted(glob.glob(os.path.join(gnn_dir, "metrics_*_seed*.json")))
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        test = d.get("test_metrics", {})
        for k in METRICS:
            v = test.get(k)
            if v is not None:
                out[k].append(float(v))
    return out, len(files)


def _collect_lgbm(lgbm_dir: str):
    only = defaultdict(list)
    stack = defaultdict(list)
    weights = []
    files = sorted(glob.glob(os.path.join(lgbm_dir, "metrics_seed*.json")))
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        for k in METRICS:
            v = d.get("lgbm_only", {}).get("test", {}).get(k)
            if v is not None:
                only[k].append(float(v))
            v2 = d.get("stack_with_gnn", {}).get("test", {}).get(k)
            if v2 is not None:
                stack[k].append(float(v2))
        if "stack_with_gnn" in d and "lgbm_weight" in d["stack_with_gnn"]:
            weights.append(float(d["stack_with_gnn"]["lgbm_weight"]))
    return only, stack, weights, len(files)


def _fmt(arr):
    if not arr:
        return "      N/A      "
    m = np.mean(arr)
    s = np.std(arr)
    return f"{m:.4f} ± {s:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-dir", default="outputs/benchmark/CHEB",
                   help="GNN metrics JSON 폴더 (FINAL = outputs/benchmark/CHEB)")
    p.add_argument("--lgbm-dir", default="outputs/lgbm",
                   help="LGBM metrics JSON 폴더")
    args = p.parse_args()

    gnn, n_gnn = _collect_gnn(args.gnn_dir)
    lgbm_only, stack, weights, n_lgbm = _collect_lgbm(args.lgbm_dir)

    print("\n" + "=" * 88)
    print(f"  5-Seed Aggregate — Test Metrics (mean ± std)")
    print(f"  GNN  files: {n_gnn}  ({args.gnn_dir})")
    print(f"  LGBM files: {n_lgbm}  ({args.lgbm_dir})")
    print("=" * 88)
    header = f"  {'Metric':<14} | {'GNN (FINAL)':<19} | {'LGBM only':<19} | {'LGBM + GNN Stack':<19}"
    print(header)
    print("  " + "-" * 86)
    for k in METRICS:
        print(f"  {k:<14} | {_fmt(gnn[k]):<19} | {_fmt(lgbm_only[k]):<19} | {_fmt(stack[k]):<19}")
    print("=" * 88)

    if weights:
        print(f"\n  Best blend weight w(LGBM) per seed: {[f'{w:.2f}' for w in weights]}")
        print(f"  → mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")
        print(f"  (w=1 → LGBM 단독, w=0 → GNN 단독, 중간이면 두 모델이 서로 보완)")

    # Stacking 효과 요약
    if gnn["pr_auc"] and stack["pr_auc"]:
        g = float(np.mean(gnn["pr_auc"]))
        s = float(np.mean(stack["pr_auc"]))
        delta = s - g
        sign = "+" if delta >= 0 else ""
        print(f"\n  Stacking lift (test PR-AUC): {g:.4f} -> {s:.4f}  ({sign}{delta:.4f})")


if __name__ == "__main__":
    main()
