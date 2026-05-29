"""
Level-2 Stacked Ensemble — LGBM + XGBoost + CatBoost + GNN.

Level 1 (base learners):
  - LightGBM  : leakage-safe behavioral features (build_features 재사용)
  - XGBoost   : 같은 피처로 학습
  - CatBoost  : 같은 피처로 학습
  - GNN       : 이미 저장된 valid/test 확률 npy 로드

Level 2 (meta-learner):
  - Logistic Regression (기본) 또는 얕은 XGBoost
  - 입력: [lgbm_v, xgb_v, cat_v, gnn_v] (valid 에서 학습)
  - 출력: test blend 확률 → 최종 PR-AUC / Macro-F1

설계 원칙:
  - 모든 행동 피처 aggregate 는 train 만으로 (lgbm_stacking 의 build_features 재사용 → 누수 차단)
  - meta-learner 는 valid 예측만으로 학습 (test 정보 한 번도 미접촉)
  - GNN 은 base-learner 단계에선 *고정 입력*. 5-fold OOF 재학습 없음 (학습 비용 큼)

사용:
  python -m src.training.stacked_ensemble --seed 42
      --gnn-probs-valid outputs/benchmark/CHEB/probs_valid_seed42.npy
      --gnn-probs-test  outputs/benchmark/CHEB/probs_test_seed42.npy
      [--meta logreg|xgb]   (기본 logreg)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import lightgbm as lgb

try:
    import xgboost as xgb_module
except ImportError as e:
    raise ImportError("pip install xgboost") from e

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise ImportError("pip install catboost") from e

from src.training.lgbm_stacking import (
    DEFAULT_PARAMS as LGBM_PARAMS,
    build_features,
    compute_metrics,
    find_best_threshold,
)


XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    verbosity=0,
)

CAT_PARAMS = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    eval_metric="PRAUC",
    early_stopping_rounds=100,
    verbose=0,
    allow_writing_files=False,
)


def _train_lgbm(X_tr, y_tr, X_va, y_va, seed):
    params = {**LGBM_PARAMS, "seed": seed}
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va)
    model = lgb.train(
        params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    return model


def _train_xgb(X_tr, y_tr, X_va, y_va, seed):
    params = {**XGB_PARAMS, "seed": seed}
    dtrain = xgb_module.DMatrix(X_tr, label=y_tr)
    dvalid = xgb_module.DMatrix(X_va, label=y_va)
    model = xgb_module.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100, verbose_eval=0,
    )
    return model


def _train_cat(X_tr, y_tr, X_va, y_va, seed):
    model = CatBoostClassifier(**CAT_PARAMS, random_seed=seed)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return model


def _predict_xgb(model, X):
    return model.predict(xgb_module.DMatrix(X), iteration_range=(0, model.best_iteration + 1))


def _score(y, p, label):
    s = average_precision_score(y, p)
    print(f"  [{label:5s}] valid PR-AUC = {s:.4f}")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/sampled_reviews.csv")
    p.add_argument("--out-dir", default="outputs/stacked")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gnn-probs-valid", required=True)
    p.add_argument("--gnn-probs-test", required=True)
    p.add_argument("--meta", default="logreg", choices=["logreg", "xgb"],
                   help="Level-2 meta-learner: logreg (기본) or xgb")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Load] {args.data}")
    df = pd.read_csv(args.data)
    train_mask = (df["split"] == "train").to_numpy()
    valid_mask = (df["split"] == "valid").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    feats = build_features(df, train_mask)
    y = df["label"].to_numpy()

    X_train = feats.loc[train_mask].to_numpy()
    X_valid = feats.loc[valid_mask].to_numpy()
    X_test = feats.loc[test_mask].to_numpy()
    y_train = y[train_mask]
    y_valid = y[valid_mask]
    y_test = y[test_mask]

    # --- Level 1 -----------------------------------------------------------
    print("\n=== Level 1: LightGBM ===")
    lgbm = _train_lgbm(X_train, y_train, X_valid, y_valid, args.seed)
    lgbm_valid = lgbm.predict(X_valid, num_iteration=lgbm.best_iteration)
    lgbm_test = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)
    _score(y_valid, lgbm_valid, "LGBM")

    print("\n=== Level 1: XGBoost ===")
    xgbm = _train_xgb(X_train, y_train, X_valid, y_valid, args.seed)
    xgb_valid = _predict_xgb(xgbm, X_valid)
    xgb_test = _predict_xgb(xgbm, X_test)
    _score(y_valid, xgb_valid, "XGB")

    print("\n=== Level 1: CatBoost ===")
    catm = _train_cat(X_train, y_train, X_valid, y_valid, args.seed)
    cat_valid = catm.predict_proba(X_valid)[:, 1]
    cat_test = catm.predict_proba(X_test)[:, 1]
    _score(y_valid, cat_valid, "Cat")

    print("\n=== Level 1: GNN (loaded) ===")
    gnn_valid = np.load(args.gnn_probs_valid)
    gnn_test = np.load(args.gnn_probs_test)
    assert len(gnn_valid) == len(y_valid), f"GNN valid len mismatch: {len(gnn_valid)} vs {len(y_valid)}"
    assert len(gnn_test) == len(y_test), f"GNN test  len mismatch: {len(gnn_test)} vs {len(y_test)}"
    _score(y_valid, gnn_valid, "GNN")

    # --- Level 2: meta-learner --------------------------------------------
    print("\n=== Level 2: Meta-learner ===")
    base_names = ["lgbm", "xgb", "cat", "gnn"]
    meta_X_valid = np.column_stack([lgbm_valid, xgb_valid, cat_valid, gnn_valid])
    meta_X_test = np.column_stack([lgbm_test, xgb_test, cat_test, gnn_test])

    if args.meta == "logreg":
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
        meta.fit(meta_X_valid, y_valid)
        valid_blend = meta.predict_proba(meta_X_valid)[:, 1]
        test_blend = meta.predict_proba(meta_X_test)[:, 1]
        meta_coefs = {n: float(c) for n, c in zip(base_names, meta.coef_[0])}
        print(f"  Meta (LogReg) coefs: {meta_coefs}")
        print(f"  Meta intercept: {float(meta.intercept_[0]):.4f}")
    else:
        # shallow XGB meta
        dtr = xgb_module.DMatrix(meta_X_valid, label=y_valid)
        meta = xgb_module.train(
            {**XGB_PARAMS, "seed": args.seed, "max_depth": 3, "learning_rate": 0.1},
            dtr, num_boost_round=100,
        )
        valid_blend = meta.predict(xgb_module.DMatrix(meta_X_valid))
        test_blend = meta.predict(xgb_module.DMatrix(meta_X_test))
        meta_coefs = None

    best_t, best_f1 = find_best_threshold(y_valid, valid_blend)
    valid_metrics = compute_metrics(y_valid, valid_blend, threshold=best_t)
    test_metrics = compute_metrics(y_test, test_blend, threshold=best_t)

    print(f"\n[Stacked-Final] valid PR-AUC = {valid_metrics['pr_auc']:.4f}  macro-F1 = {valid_metrics['macro_f1']:.4f}")
    print(f"[Stacked-Final] test  PR-AUC = {test_metrics['pr_auc']:.4f}   macro-F1 = {test_metrics['macro_f1']:.4f}")

    out = {
        "seed": args.seed,
        "meta": args.meta,
        "level1_test_pr_auc": {
            "lgbm": float(average_precision_score(y_test, lgbm_test)),
            "xgb": float(average_precision_score(y_test, xgb_test)),
            "cat": float(average_precision_score(y_test, cat_test)),
            "gnn": float(average_precision_score(y_test, gnn_test)),
        },
        "level1_valid_pr_auc": {
            "lgbm": float(average_precision_score(y_valid, lgbm_valid)),
            "xgb": float(average_precision_score(y_valid, xgb_valid)),
            "cat": float(average_precision_score(y_valid, cat_valid)),
            "gnn": float(average_precision_score(y_valid, gnn_valid)),
        },
        "level2_test_metrics": test_metrics,
        "level2_valid_metrics": valid_metrics,
        "best_threshold": best_t,
        "meta_coefs": meta_coefs,
    }

    metrics_path = os.path.join(args.out_dir, f"metrics_seed{args.seed}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.out_dir, f"probs_test_seed{args.seed}.npy"), test_blend)
    print(f"\n[Save] {metrics_path}")


if __name__ == "__main__":
    main()
