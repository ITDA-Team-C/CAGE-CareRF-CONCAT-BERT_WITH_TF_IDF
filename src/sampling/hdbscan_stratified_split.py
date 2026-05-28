"""
HDBSCAN 기반 의미 층화 분할 (Semantic Stratified Split for Anomaly Detection)

핵심 아이디어:
  1) 임베딩을 HDBSCAN으로 군집화 → 클러스터 ID(노이즈는 -1)
  2) stratify key = (label, cluster_id) → 라벨 분포 + 의미 분포를 동시에 정렬
  3) 노이즈 포인트는 별도 stratum으로 처리해서 분포 보존
  4) 분할 후 JS-divergence / 라벨 비율 / 노이즈 비율로 분할 품질 검증

사용처: 이상 탐지의 (정상/이상) 이진 라벨 + 의미 군집을 함께 맞추고 싶을 때.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import hdbscan
except ImportError as e:
    raise ImportError("pip install hdbscan") from e


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _cluster_distribution(cluster_ids: np.ndarray, all_clusters: np.ndarray) -> np.ndarray:
    counts = np.array([(cluster_ids == c).sum() for c in all_clusters], dtype=float)
    return counts / max(counts.sum(), 1.0)


def hdbscan_cluster(
    embeddings: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    """HDBSCAN으로 군집화. 반환값: cluster_id 배열 (노이즈 = -1)."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1,
    )
    return clusterer.fit_predict(embeddings)


def _stratify_key(labels: np.ndarray, clusters: np.ndarray, min_per_stratum: int = 2) -> np.ndarray:
    """(label, cluster_id) 결합 키. 너무 작은 stratum은 'rare'로 합쳐서 split 실패 방지."""
    keys = np.array([f"{l}_{c}" for l, c in zip(labels, clusters)])
    uniq, cnt = np.unique(keys, return_counts=True)
    rare = set(uniq[cnt < min_per_stratum])
    # rare stratum은 라벨별 'rare' 버킷으로 통합 → 라벨 분포는 보존
    keys = np.array([f"{k.split('_')[0]}_rare" if k in rare else k for k in keys])
    return keys


def hdbscan_stratified_split(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str = "label",
    test_size: float = 0.15,
    valid_size: float = 0.15,
    min_cluster_size: int = 30,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    HDBSCAN 의미 클러스터 + 라벨 동시 층화로 train/valid/test 분할.

    Parameters
    ----------
    df : 원본 DataFrame (label_col 포함). 인덱스는 embeddings 행 순서와 일치해야 함.
    embeddings : (N, D) numpy 배열 (SBERT 등).
    label_col : 라벨 컬럼명 (이상 탐지: 0=정상, 1=이상).
    test_size, valid_size : 비율.

    Returns
    -------
    train_df, valid_df, test_df, report
        report에는 분할 품질 지표(JS-divergence, 라벨 비율, 노이즈 비율 등)가 들어감.
    """
    assert len(df) == len(embeddings), "df와 embeddings 길이가 다릅니다."
    df = df.reset_index(drop=True).copy()

    labels = df[label_col].to_numpy()
    cluster_ids = hdbscan_cluster(embeddings, min_cluster_size=min_cluster_size, random_state=random_state)
    df["_cluster_id"] = cluster_ids

    if verbose:
        n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_noise = int((cluster_ids == -1).sum())
        print(f"[HDBSCAN] clusters={n_clusters}, noise={n_noise} ({n_noise/len(df):.2%})")

    strat_key = _stratify_key(labels, cluster_ids)

    # 1단계: train+valid vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trv_idx, test_idx = next(sss1.split(np.zeros(len(df)), strat_key))

    # 2단계: train vs valid (valid_size는 전체 기준 → trainvalid 내 비율로 환산)
    valid_ratio_within = valid_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio_within, random_state=random_state)
    train_rel, valid_rel = next(sss2.split(np.zeros(len(trv_idx)), strat_key[trv_idx]))
    train_idx = trv_idx[train_rel]
    valid_idx = trv_idx[valid_rel]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 품질 검증
    all_clusters = np.unique(cluster_ids)
    p_train = _cluster_distribution(train_df["_cluster_id"].to_numpy(), all_clusters)
    p_valid = _cluster_distribution(valid_df["_cluster_id"].to_numpy(), all_clusters)
    p_test = _cluster_distribution(test_df["_cluster_id"].to_numpy(), all_clusters)

    report = {
        "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        "anomaly_ratio": {
            "train": float((train_df[label_col] == 1).mean()),
            "valid": float((valid_df[label_col] == 1).mean()),
            "test": float((test_df[label_col] == 1).mean()),
        },
        "noise_ratio": {
            "train": float((train_df["_cluster_id"] == -1).mean()),
            "valid": float((valid_df["_cluster_id"] == -1).mean()),
            "test": float((test_df["_cluster_id"] == -1).mean()),
        },
        "cluster_js_divergence": {
            "train_vs_valid": _js_divergence(p_train, p_valid),
            "train_vs_test": _js_divergence(p_train, p_test),
            "valid_vs_test": _js_divergence(p_valid, p_test),
        },
        "n_clusters": int(len(all_clusters) - (1 if -1 in all_clusters else 0)),
    }

    if verbose:
        print(f"[Split sizes] {report['sizes']}")
        print(f"[Anomaly ratio] {report['anomaly_ratio']}")
        print(f"[Cluster JS-div] {report['cluster_js_divergence']}")
        print("  (JS-div < 0.05 면 거의 동일, < 0.1 양호, > 0.2 편향 의심)")

    return train_df, valid_df, test_df, report


if __name__ == "__main__":
    # 사용 예시
    rng = np.random.default_rng(0)
    N, D = 2000, 64
    X = rng.normal(size=(N, D))
    # 가짜 라벨 (5% 이상)
    y = (rng.random(N) < 0.05).astype(int)
    df = pd.DataFrame({"id": np.arange(N), "label": y})

    tr, va, te, rep = hdbscan_stratified_split(
        df, X, label_col="label", test_size=0.15, valid_size=0.15, min_cluster_size=30
    )
