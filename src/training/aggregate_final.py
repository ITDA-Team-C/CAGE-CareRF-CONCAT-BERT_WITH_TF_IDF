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


def _collect_stacked(stacked_dir: str):
    meta = defaultdict(list)
    l1 = defaultdict(lambda: defaultdict(list))   # l1[model][metric] -> list
    coefs_history = []
    files = sorted(glob.glob(os.path.join(stacked_dir, "metrics_seed*.json")))
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        for k in METRICS:
            v = d.get("level2_test_metrics", {}).get(k)
            if v is not None:
                meta[k].append(float(v))
        for base, score in (d.get("level1_test_pr_auc", {}) or {}).items():
            l1[base]["pr_auc"].append(float(score))
        if d.get("meta_coefs"):
            coefs_history.append(d["meta_coefs"])
    return meta, l1, coefs_history, len(files)


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
    p.add_argument("--stacked-dir", default="outputs/stacked",
                   help="Level-2 stacked ensemble metrics JSON 폴더")
    args = p.parse_args()

    gnn, n_gnn = _collect_gnn(args.gnn_dir)
    lgbm_only, stack, weights, n_lgbm = _collect_lgbm(args.lgbm_dir)
    meta, l1, coefs, n_stk = _collect_stacked(args.stacked_dir)

    print("\n" + "=" * 110)
    print(f"  5-Seed Aggregate — Test Metrics (mean ± std)")
    print(f"  GNN     files: {n_gnn}  ({args.gnn_dir})")
    print(f"  LGBM    files: {n_lgbm}  ({args.lgbm_dir})")
    print(f"  Stacked files: {n_stk}  ({args.stacked_dir})")
    print("=" * 110)
    header = (f"  {'Metric':<12} | {'GNN (FINAL)':<17} | {'LGBM only':<17} | "
              f"{'LGBM+GNN Blend':<17} | {'L2 Meta Ensemble':<17}")
    print(header)
    print("  " + "-" * 108)
    for k in METRICS:
        print(f"  {k:<12} | {_fmt(gnn[k]):<17} | {_fmt(lgbm_only[k]):<17} | "
              f"{_fmt(stack[k]):<17} | {_fmt(meta[k]):<17}")
    print("=" * 110)

    if weights:
        print(f"\n  Best blend weight w(LGBM) per seed: {[f'{w:.2f}' for w in weights]}")
        print(f"  → mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")
        print(f"  (w=1 → LGBM 단독, w=0 → GNN 단독, 중간이면 두 모델이 서로 보완)")

    # Stacking 효과 요약 (weighted blend vs GNN)
    if gnn["pr_auc"] and stack["pr_auc"]:
        g = float(np.mean(gnn["pr_auc"]))
        s = float(np.mean(stack["pr_auc"]))
        delta = s - g
        sign = "+" if delta >= 0 else ""
        print(f"\n  Blend lift  (test PR-AUC, GNN→LGBM+GNN):  {g:.4f} -> {s:.4f}  ({sign}{delta:.4f})")

    # Level-2 Meta 효과 요약
    if gnn["pr_auc"] and meta["pr_auc"]:
        g = float(np.mean(gnn["pr_auc"]))
        m = float(np.mean(meta["pr_auc"]))
        delta = m - g
        sign = "+" if delta >= 0 else ""
        print(f"  Meta  lift  (test PR-AUC, GNN→L2-Meta):    {g:.4f} -> {m:.4f}  ({sign}{delta:.4f})")

    # Level-1 base learner 단독 비교 (stacked 출력에서)
    if l1:
        print("\n  Level-1 base learners (test PR-AUC, mean):")
        for base, ks in l1.items():
            if ks["pr_auc"]:
                print(f"    {base:>5s}: {np.mean(ks['pr_auc']):.4f} ± {np.std(ks['pr_auc']):.4f}")

    # Meta-learner 가중치 평균 (어느 base 가 가장 의존되는지)
    if coefs:
        keys = list(coefs[0].keys())
        print("\n  Meta-learner coef (mean across seeds):")
        for k in keys:
            vals = [c.get(k, 0.0) for c in coefs]
            print(f"    {k:>5s}: {np.mean(vals):+.3f}  (std {np.std(vals):.3f})")


if __name__ == "__main__":
    main()
