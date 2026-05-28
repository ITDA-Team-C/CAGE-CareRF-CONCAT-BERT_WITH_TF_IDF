"""
Hand-crafted Behavioral Features + LightGBM (Option 1 of non-GNN uplift).

설계 원칙:
  - 모든 aggregate (user/prod 통계, fraud-rate prior 등)는 **train split 에서만** 계산.
    test/valid 라벨이 피처에 흘러들어가지 못하도록 차단 (data-leakage 방지).
  - 피처 그룹:
      * self      : 리뷰 자체 (별점, 텍스트 길이, 느낌표·대문자 등)
      * user_agg  : 사용자 행동 통계 (train 기반)
      * prod_agg  : 상품 통계 (train 기반)
      * burst     : 시간 윈도우 burst (같은 prod ±N일 내 리뷰 수)
      * relative  : 이 리뷰가 user/prod 평균에서 얼마나 벗어났는가
  - GNN 확률과 stacking 은 단순 weighted blend (valid 에서 weight 탐색).

사용법:
    python -m src.training.lgbm_stacking
        --data data/processed/sampled_reviews.csv
        --gnn-probs outputs/cage_rf_gnn/predictions_test_seed42.npy   (옵션)
        --blend-weight 0.5                                            (옵션, 미지정 시 valid 에서 그리드 탐색)
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("pip install lightgbm") from e


# ---------------------------------------------------------------------------
# Feature builders — 모든 aggregate 는 train_df 만으로 만들어 *전체* 에 매핑한다.
# ---------------------------------------------------------------------------

def _self_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)
    out["rating"] = df["rating"].astype(float)
    out["rating_is_extreme"] = ((df["rating"] == 1) | (df["rating"] == 5)).astype(int)
    out["text_len"] = text.str.len()
    out["text_n_words"] = text.str.split().str.len().fillna(0)
    out["text_n_exclaim"] = text.str.count("!")
    out["text_n_question"] = text.str.count(r"\?")
    out["text_caps_ratio"] = text.apply(
        lambda s: sum(c.isupper() for c in s) / max(len(s), 1)
    )
    out["text_avg_word_len"] = out["text_len"] / out["text_n_words"].replace(0, 1)
    # TTR (type-token ratio): 어휘 다양성. 사기 리뷰는 템플릿이라 낮은 경향
    out["text_ttr"] = text.apply(
        lambda s: len(set(s.lower().split())) / max(len(s.split()), 1)
    )
    # 시간 피처
    dt = pd.to_datetime(df["date"], errors="coerce")
    out["dow"] = dt.dt.dayofweek.fillna(-1).astype(int)
    out["month"] = dt.dt.month.fillna(-1).astype(int)
    out["year"] = dt.dt.year.fillna(-1).astype(int)
    return out


def _user_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    g = train_df.groupby("user_id")
    out = pd.DataFrame({
        "user_n_reviews": g.size(),
        "user_rating_mean": g["rating"].mean(),
        "user_rating_std": g["rating"].std().fillna(0),
        "user_rating_extreme_ratio": g["rating"].apply(
            lambda x: ((x == 1) | (x == 5)).mean()
        ),
        "user_text_len_mean": g["text"].apply(lambda x: x.fillna("").str.len().mean()),
        "user_text_len_std": g["text"].apply(
            lambda x: x.fillna("").str.len().std()
        ).fillna(0),
        "user_unique_prods": g["prod_id"].nunique(),
    })
    out["user_review_per_prod"] = out["user_n_reviews"] / out["user_unique_prods"].replace(0, 1)
    return out


def _prod_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    g = train_df.groupby("prod_id")
    out = pd.DataFrame({
        "prod_n_reviews": g.size(),
        "prod_rating_mean": g["rating"].mean(),
        "prod_rating_std": g["rating"].std().fillna(0),
        "prod_rating_skew": g["rating"].apply(lambda x: x.skew()).fillna(0),
        "prod_extreme_ratio": g["rating"].apply(
            lambda x: ((x == 1) | (x == 5)).mean()
        ),
        "prod_unique_users": g["user_id"].nunique(),
    })
    out["prod_review_per_user"] = out["prod_n_reviews"] / out["prod_unique_users"].replace(0, 1)
    return out


def _burst_features(df: pd.DataFrame, prod_window_days: int = 7) -> pd.DataFrame:
    """
    각 리뷰에 대해 같은 prod_id 의 [date ± window] 안에 들어가는 리뷰 수.
    binary-search 로 O(N log N) — 40k~50k 규모에서 충분히 빠름.
    """
    dt = pd.to_datetime(df["date"], errors="coerce")
    prod_ids = df["prod_id"].to_numpy()
    burst_counts = np.zeros(len(df), dtype=int)
    window_ns = pd.Timedelta(days=prod_window_days).value

    # prod_id 별 정렬된 (date_ns, original_index) 리스트
    df_idx = np.arange(len(df))
    date_ns = dt.values.astype("datetime64[ns]").astype("int64")

    order = np.lexsort((date_ns, prod_ids))
    sorted_prods = prod_ids[order]
    sorted_dates = date_ns[order]
    sorted_orig = df_idx[order]

    # prod_id 별 시작·끝 인덱스 찾기 (정렬됐으므로 연속 구간)
    n = len(sorted_prods)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_prods[end] == sorted_prods[start]:
            end += 1
        # [start, end) 같은 prod
        dates_chunk = sorted_dates[start:end]
        for k in range(end - start):
            d = dates_chunk[k]
            lo = bisect.bisect_left(dates_chunk, d - window_ns)
            hi = bisect.bisect_right(dates_chunk, d + window_ns)
            burst_counts[sorted_orig[start + k]] = hi - lo
        start = end

    out = pd.DataFrame({f"burst_prod_{prod_window_days}d": burst_counts}, index=df.index)
    return out


def build_features(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """
    전체 df 에 대해 피처 매트릭스 생성. user/prod aggregate 는 train_mask=True 만으로 계산.

    Returns
    -------
    DataFrame, shape (N, n_features). 원본 df 의 index 보존.
    """
    train_df = df.loc[train_mask].copy()

    print(f"[Features] self / text / time...")
    self_f = _self_features(df)

    print(f"[Features] user aggregates (train-only) ...")
    user_agg = _user_aggregates(train_df)
    user_mapped = df["user_id"].map(user_agg.to_dict(orient="index")).apply(pd.Series)
    user_mapped.index = df.index
    # 매핑 시 train 에 없던 사용자(test only user) → NaN → 0 + 별도 binary flag
    user_mapped["user_unseen_in_train"] = user_mapped["user_n_reviews"].isna().astype(int)
    user_mapped = user_mapped.fillna(0)

    print(f"[Features] prod aggregates (train-only) ...")
    prod_agg = _prod_aggregates(train_df)
    prod_mapped = df["prod_id"].map(prod_agg.to_dict(orient="index")).apply(pd.Series)
    prod_mapped.index = df.index
    prod_mapped["prod_unseen_in_train"] = prod_mapped["prod_n_reviews"].isna().astype(int)
    prod_mapped = prod_mapped.fillna(0)

    print(f"[Features] burst window ...")
    burst_f = _burst_features(df, prod_window_days=7)

    # Relative 피처: 이 리뷰가 자기 user/prod 의 평균에서 얼마나 벗어났는가
    rel = pd.DataFrame(index=df.index)
    rel["rating_vs_user_mean"] = self_f["rating"] - user_mapped["user_rating_mean"]
    rel["rating_vs_prod_mean"] = self_f["rating"] - prod_mapped["prod_rating_mean"]
    rel["len_vs_user_mean"] = self_f["text_len"] - user_mapped["user_text_len_mean"]

    feats = pd.concat([self_f, user_mapped, prod_mapped, burst_f, rel], axis=1)
    print(f"[Features] total shape: {feats.shape}")
    return feats


# ---------------------------------------------------------------------------
# LightGBM 학습 + 평가
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
)


def train_lgbm(X_train, y_train, X_valid, y_valid, params=None, num_boost_round=2000,
               early_stopping_rounds=100, seed=42):
    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def compute_metrics(y_true, y_score, threshold=None):
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    if threshold is None:
        # valid 에서 F1 최대화 threshold 를 찾는 게 좋지만, 여기서는 0.5 기본
        threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "macro_f1": float(macro_f1),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, y_score):
    """valid 에서 macro-F1 을 최대화하는 threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds[::max(1, len(thresholds) // 100)]:
        f1 = f1_score(y_true, (y_score >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# ---------------------------------------------------------------------------
# GNN 확률과 stacking
# ---------------------------------------------------------------------------

def blend_with_gnn(lgbm_probs, gnn_probs, y_valid_lgbm, y_valid_gnn,
                   weights=None):
    """
    Weighted average blend. weights 미지정 시 valid 에서 PR-AUC 최대화하는 w 탐색.
    Returns: (blended_test_probs, best_weight_for_lgbm)
    """
    if weights is None:
        weights = np.arange(0.0, 1.05, 0.05)
    best_w, best_score = 0.5, -1.0
    for w in weights:
        blend_valid = w * y_valid_lgbm + (1 - w) * y_valid_gnn
        # 외부에서 valid 의 y_true 와 score 만 받으면 PR-AUC 계산 가능 — 여기선 외부 호출자에 위임
        # 이 함수는 단순 평균 계산만 제공
        pass
    raise NotImplementedError("이 함수는 wrapper 에서 valid y_true 와 함께 호출하세요.")


def stack_blend_search(valid_y, valid_lgbm, valid_gnn, test_lgbm, test_gnn,
                       weights=None):
    """
    valid 에서 LightGBM:GNN 가중치 w 를 PR-AUC 로 그리드 탐색 → test 에 적용.
    """
    if weights is None:
        weights = np.arange(0.0, 1.05, 0.05)
    best_w, best_score = 0.5, -1.0
    for w in weights:
        s = w * valid_lgbm + (1 - w) * valid_gnn
        score = average_precision_score(valid_y, s)
        if score > best_score:
            best_w, best_score = float(w), float(score)
    test_blend = best_w * test_lgbm + (1 - best_w) * test_gnn
    return test_blend, best_w, best_score


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/sampled_reviews.csv")
    p.add_argument("--out-dir", default="outputs/lgbm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gnn-probs-valid", default=None,
                   help="GNN valid 확률 npy 경로 (stacking 옵션)")
    p.add_argument("--gnn-probs-test", default=None,
                   help="GNN test 확률 npy 경로 (stacking 옵션)")
    args = p.parse_args(args)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Load] {args.data}")
    df = pd.read_csv(args.data)
    assert "split" in df.columns and "label" in df.columns, "split/label 컬럼 필요"

    train_mask = (df["split"] == "train").to_numpy()
    valid_mask = (df["split"] == "valid").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    print(f"  train={train_mask.sum()}  valid={valid_mask.sum()}  test={test_mask.sum()}")
    print(f"  fraud ratio — train={df.loc[train_mask, 'label'].mean():.4f}, "
          f"valid={df.loc[valid_mask, 'label'].mean():.4f}, "
          f"test={df.loc[test_mask, 'label'].mean():.4f}")

    feats = build_features(df, train_mask)
    y = df["label"].to_numpy()

    X_train = feats.loc[train_mask].to_numpy()
    X_valid = feats.loc[valid_mask].to_numpy()
    X_test = feats.loc[test_mask].to_numpy()
    y_train = y[train_mask]
    y_valid = y[valid_mask]
    y_test = y[test_mask]

    print(f"\n[Train] LightGBM (seed={args.seed})...")
    model = train_lgbm(X_train, y_train, X_valid, y_valid, seed=args.seed)

    valid_probs = model.predict(X_valid, num_iteration=model.best_iteration)
    test_probs = model.predict(X_test, num_iteration=model.best_iteration)

    best_t, best_f1 = find_best_threshold(y_valid, valid_probs)
    print(f"\n[Threshold] valid best macro-F1 threshold = {best_t:.4f} (F1={best_f1:.4f})")

    valid_metrics = compute_metrics(y_valid, valid_probs, threshold=best_t)
    test_metrics = compute_metrics(y_test, test_probs, threshold=best_t)
    print(f"[LGBM-only] valid  PR-AUC={valid_metrics['pr_auc']:.4f}  macro-F1={valid_metrics['macro_f1']:.4f}")
    print(f"[LGBM-only] test   PR-AUC={test_metrics['pr_auc']:.4f}   macro-F1={test_metrics['macro_f1']:.4f}")

    # 저장
    np.save(os.path.join(args.out_dir, f"probs_valid_seed{args.seed}.npy"), valid_probs)
    np.save(os.path.join(args.out_dir, f"probs_test_seed{args.seed}.npy"), test_probs)
    pd.DataFrame({
        "feature": feats.columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).to_csv(
        os.path.join(args.out_dir, f"feature_importance_seed{args.seed}.csv"),
        index=False,
    )

    out_metrics = {
        "lgbm_only": {"valid": valid_metrics, "test": test_metrics, "best_threshold": best_t},
    }

    # Optional: stack with GNN
    if args.gnn_probs_valid and args.gnn_probs_test:
        print(f"\n[Stack] GNN valid={args.gnn_probs_valid}  test={args.gnn_probs_test}")
        gnn_valid = np.load(args.gnn_probs_valid)
        gnn_test = np.load(args.gnn_probs_test)
        assert len(gnn_valid) == len(y_valid), \
            f"GNN valid len {len(gnn_valid)} != y_valid len {len(y_valid)}"
        assert len(gnn_test) == len(y_test), \
            f"GNN test len {len(gnn_test)} != y_test len {len(y_test)}"

        test_blend, w, valid_blend_score = stack_blend_search(
            y_valid, valid_probs, gnn_valid, test_probs, gnn_test,
        )
        valid_blend = w * valid_probs + (1 - w) * gnn_valid
        bt2, bf2 = find_best_threshold(y_valid, valid_blend)
        stack_test_metrics = compute_metrics(y_test, test_blend, threshold=bt2)
        print(f"[Stack] best w(LGBM)={w:.2f}, valid PR-AUC={valid_blend_score:.4f}")
        print(f"[Stack] test PR-AUC={stack_test_metrics['pr_auc']:.4f}  "
              f"macro-F1={stack_test_metrics['macro_f1']:.4f}")
        np.save(os.path.join(args.out_dir, f"probs_test_stack_seed{args.seed}.npy"), test_blend)
        out_metrics["stack_with_gnn"] = {
            "lgbm_weight": w,
            "test": stack_test_metrics,
            "best_threshold": bt2,
        }

    metrics_path = os.path.join(args.out_dir, f"metrics_seed{args.seed}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Save] {metrics_path}")


if __name__ == "__main__":
    main()
