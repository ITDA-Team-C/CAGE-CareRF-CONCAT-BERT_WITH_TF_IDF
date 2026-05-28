"""
3-Stage Cascade Sampling Pipeline

Stage 1 (외부 호출): Group-dense — fraud-active 그룹 시드 선택
  ↓ ~25% fraud, ~44k 노드 (sampling.group_dense_sampling 결과)
Stage 2 (semantic_filter): 의미적 필터
  - TF-IDF + TruncatedSVD + HDBSCAN 으로 텍스트 군집화
  - 클러스터별 fraud density 기반 선택적 유지
  ↓ ~30% fraud
Stage 3 (behavioral_reseed): 행동적 확장 (연결성 복원)
  - fraud 노드의 R-U-R / R-P-R 1-hop normal 노드 일부 복원
  - 학습 가능한 graph 구조 유지 (CARE filter 의 대조군 확보)
  ↓ ~28% fraud, 그래프 연결성 회복

설계 의도:
  Recall → Precision → Learnability 의 데이터 큐레이션 순서.
  단순 fraud-rich 추출이 아니라 GNN 학습에 최적화된 큐레이션.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _tfidf_svd_embed(texts: pd.Series, dim: int = 64, random_state: int = 42) -> np.ndarray:
    """TF-IDF + TruncatedSVD 로 가벼운 의미 임베딩 생성 (SBERT 없이 sampling 단계 처리)."""
    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts.fillna("").astype(str))
    n_components = min(dim, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    emb = svd.fit_transform(X)
    return emb


def _hdbscan_cluster(emb: np.ndarray, min_cluster_size: int = 50, random_state: int = 42) -> np.ndarray:
    """
    HDBSCAN 시도 → 결과가 0개 클러스터(전부 noise)면 MiniBatchKMeans 폴백.
    TF-IDF+SVD 임베딩은 밀도가 균일해서 HDBSCAN이 클러스터를 못 찾는 경우가 있음.
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "cascade Stage 2 requires hdbscan. Install: pip install hdbscan"
        ) from e
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",  # leaf 가 더 잘게 쪼개서 density 차이 드러남
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(emb)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float((labels == -1).mean()) if -1 in labels else 0.0

    # HDBSCAN 이 충분한 클러스터를 못 찾거나(< 5) noise 비율이 너무 높으면(> 50%) KMeans 폴백.
    # TF-IDF+SVD 임베딩은 밀도가 균일해서 HDBSCAN 보다 KMeans 가 안정적.
    if n_clusters < 5 or noise_ratio > 0.50:
        from sklearn.cluster import MiniBatchKMeans
        k = min(50, max(10, len(emb) // 500))
        print(f"  [fallback] HDBSCAN clusters={n_clusters}, noise={noise_ratio:.2%} "
              f"→ MiniBatchKMeans(k={k})")
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init="auto", batch_size=1024)
        labels = km.fit_predict(emb)
    return labels


def semantic_filter(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    high_density_threshold: float = 0.30,
    low_density_threshold: float = 0.05,
    low_density_keep_ratio: float = 0.5,
    noise_keep_ratio: float = 0.5,
    min_cluster_size: int = 50,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stage 2: HDBSCAN 의미 클러스터링 + fraud-density 기반 선택적 유지.

    - high-density (>= high_density_threshold): 클러스터 통째로 유지
    - low-density  (< low_density_threshold):  low_density_keep_ratio 만큼 다운샘플
    - 중간 밀도:                                 통째로 유지 (의미 컨텍스트 보존)
    - noise (-1):                                noise_keep_ratio 만큼 유지
    """
    if verbose:
        print("\n[Cascade Stage 2] Semantic filter (HDBSCAN)...")
        print(f"  input: n={len(df)}, fraud_ratio={(df[label_col]==1).mean():.4f}")

    emb = _tfidf_svd_embed(df[text_col], dim=64, random_state=random_state)
    cluster_ids = _hdbscan_cluster(emb, min_cluster_size=min_cluster_size, random_state=random_state)

    df = df.reset_index(drop=True).copy()
    df["_cid"] = cluster_ids

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = int((cluster_ids == -1).sum())
    if verbose:
        print(f"  HDBSCAN: clusters={n_clusters}, noise={n_noise} ({n_noise/len(df):.2%})")

    stats = df.groupby("_cid").agg(n=(label_col, "size"), nf=(label_col, "sum"))
    stats["density"] = stats["nf"] / stats["n"]

    if verbose:
        # 클러스터 fraud-density 분포 quantile 출력 → threshold 튜닝 근거
        d = stats[stats.index != -1]["density"]
        if len(d) > 0:
            print(f"  cluster density quantiles: "
                  f"p10={d.quantile(0.1):.3f}, p50={d.quantile(0.5):.3f}, "
                  f"p90={d.quantile(0.9):.3f}, max={d.max():.3f}")
            n_high = int((d >= high_density_threshold).sum())
            n_low = int((d < low_density_threshold).sum())
            print(f"  buckets: high(>={high_density_threshold})={n_high} clusters, "
                  f"low(<{low_density_threshold})={n_low} clusters, "
                  f"mid={len(d)-n_high-n_low} clusters")

    rng = np.random.default_rng(random_state)
    keep_mask = np.zeros(len(df), dtype=bool)
    n_high = n_low = n_mid = n_noise_kept = 0

    for cid, row in stats.iterrows():
        idx = np.where(cluster_ids == cid)[0]
        density = row["density"]

        if cid == -1:
            k = int(len(idx) * noise_keep_ratio)
            chosen = rng.choice(idx, size=k, replace=False) if k > 0 else idx[:0]
            n_noise_kept += len(chosen)
        elif density >= high_density_threshold:
            chosen = idx
            n_high += len(chosen)
        elif density < low_density_threshold:
            k = int(len(idx) * low_density_keep_ratio)
            chosen = rng.choice(idx, size=k, replace=False) if k > 0 else idx[:0]
            n_low += len(chosen)
        else:
            chosen = idx
            n_mid += len(chosen)

        keep_mask[chosen] = True

    out = df[keep_mask].drop(columns=["_cid"]).reset_index(drop=True)
    fr = (out[label_col] == 1).mean()

    if verbose:
        print(f"  kept high-density: {n_high}, mid: {n_mid}, "
              f"low-density (downsampled): {n_low}, noise (downsampled): {n_noise_kept}")
        print(f"  output: n={len(out)}, fraud_ratio={fr:.4f}")

    return out


def behavioral_reseed(
    stage2_df: pd.DataFrame,
    stage1_df: pd.DataFrame,
    label_col: str = "label",
    id_col: str = "review_id",
    user_col: str = "user_id",
    prod_col: str = "prod_id",
    date_col: str = "date",
    burst_window_days: int = 7,
    burst_min_reviews: int = 5,
    normal_recovery_ratio: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stage 3: Stage 2 결과 + R-U-R / burst 연결된 normal 노드 부분 복원.

    복원 대상: Stage 1 에 있지만 Stage 2 에서 빠진 normal 노드 중,
      - Stage 2 fraud 노드와 같은 user_id (R-U-R 1-hop), 또는
      - Stage 2 fraud 노드가 작성된 prod_id × burst window 안에 들어가는 노드

    normal_recovery_ratio 만큼 후보 풀에서 무작위 복원 → 학습 가능한 대조군 확보.
    """
    if verbose:
        print("\n[Cascade Stage 3] Behavioral reseed (R-U-R + burst)...")
        print(f"  stage2: n={len(stage2_df)}, fraud_ratio={(stage2_df[label_col]==1).mean():.4f}")

    fraud_s2 = stage2_df[stage2_df[label_col] == 1]
    fraud_users = set(fraud_s2[user_col].unique())
    fraud_prods = set(fraud_s2[prod_col].unique())

    # Stage 1 에 있지만 Stage 2 에서 빠진 후보 풀
    kept_ids = set(stage2_df[id_col])
    candidates = stage1_df[~stage1_df[id_col].isin(kept_ids)].copy()
    candidates = candidates[candidates[label_col] == 0]  # normal 만 복원

    # R-U-R 후보: fraud user 의 다른 normal 리뷰
    rur_mask = candidates[user_col].isin(fraud_users)

    # Burst 후보: fraud prod 의 normal 리뷰 중 burst window 안에 들어가는 것
    candidates[date_col] = pd.to_datetime(candidates[date_col], errors="coerce")
    fraud_s2 = fraud_s2.copy()
    fraud_s2[date_col] = pd.to_datetime(fraud_s2[date_col], errors="coerce")

    # 각 fraud prod 에 대해 fraud 리뷰들의 날짜 ± burst_window_days 범위 합집합
    burst_mask = pd.Series(False, index=candidates.index)
    if len(fraud_s2) > 0:
        fraud_prod_dates = fraud_s2.groupby(prod_col)[date_col].apply(list).to_dict()
        cand_in_fraud_prod = candidates[candidates[prod_col].isin(fraud_prods)]
        for prod, sub in cand_in_fraud_prod.groupby(prod_col):
            f_dates = pd.to_datetime(fraud_prod_dates.get(prod, []))
            if len(f_dates) < burst_min_reviews:
                continue
            cand_dates = sub[date_col]
            # 각 cand 의 날짜가 어떤 fraud 날짜의 ±window 안에 들어가면 True
            window = pd.Timedelta(days=burst_window_days)
            in_burst = np.zeros(len(sub), dtype=bool)
            cand_arr = cand_dates.to_numpy()
            for fd in f_dates:
                if pd.isna(fd):
                    continue
                in_burst |= (cand_arr >= (fd - window)) & (cand_arr <= (fd + window))
            burst_mask.loc[sub.index] = in_burst

    recovery_mask = rur_mask | burst_mask
    recovery_pool = candidates[recovery_mask]

    n_rur = int(rur_mask.sum())
    n_burst = int(burst_mask.sum())
    n_pool = len(recovery_pool)

    if n_pool == 0:
        if verbose:
            print("  recovery pool 비어있음 — Stage 2 결과 그대로 반환")
        out = stage2_df.copy()
    else:
        n_recover = int(n_pool * normal_recovery_ratio)
        recovered = recovery_pool.sample(n=n_recover, random_state=random_state)
        out = pd.concat([stage2_df, recovered], axis=0, ignore_index=True)

    # node_id 재부여
    out = out.reset_index(drop=True)
    out["node_id"] = np.arange(len(out))

    fr = (out[label_col] == 1).mean()
    if verbose:
        print(f"  candidates: rur={n_rur}, burst={n_burst}, pool(union)={n_pool}")
        print(f"  recovered: {len(out) - len(stage2_df)} (ratio={normal_recovery_ratio})")
        print(f"  output: n={len(out)}, fraud_ratio={fr:.4f}")

    return out


def run_cascade(
    stage1_df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    id_col: str = "review_id",
    user_col: str = "user_id",
    prod_col: str = "prod_id",
    date_col: str = "date",
    # Stage 2 params
    s2_high_density: float = 0.30,
    s2_low_density: float = 0.05,
    s2_low_keep_ratio: float = 0.5,
    s2_noise_keep_ratio: float = 0.5,
    s2_min_cluster_size: int = 50,
    # Stage 3 params
    s3_burst_window_days: int = 7,
    s3_burst_min_reviews: int = 5,
    s3_normal_recovery_ratio: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Stage 1 결과 → Stage 2 (semantic) → Stage 3 (behavioral) 전체 캐스케이드."""
    if verbose:
        print("\n" + "=" * 70)
        print("[Cascade Pipeline] 3-Stage Sampling 시작")
        print("=" * 70)
        print(f"[Stage 1] (외부 입력) n={len(stage1_df)}, "
              f"fraud_ratio={(stage1_df[label_col]==1).mean():.4f}")

    stage2_df = semantic_filter(
        stage1_df,
        text_col=text_col, label_col=label_col,
        high_density_threshold=s2_high_density,
        low_density_threshold=s2_low_density,
        low_density_keep_ratio=s2_low_keep_ratio,
        noise_keep_ratio=s2_noise_keep_ratio,
        min_cluster_size=s2_min_cluster_size,
        random_state=random_state,
        verbose=verbose,
    )

    stage3_df = behavioral_reseed(
        stage2_df, stage1_df,
        label_col=label_col, id_col=id_col,
        user_col=user_col, prod_col=prod_col, date_col=date_col,
        burst_window_days=s3_burst_window_days,
        burst_min_reviews=s3_burst_min_reviews,
        normal_recovery_ratio=s3_normal_recovery_ratio,
        random_state=random_state,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("[Cascade Pipeline] 완료")
        print("=" * 70)
        s1n, s2n, s3n = len(stage1_df), len(stage2_df), len(stage3_df)
        s1f = (stage1_df[label_col]==1).mean()
        s2f = (stage2_df[label_col]==1).mean()
        s3f = (stage3_df[label_col]==1).mean()
        print(f"  Stage 1: n={s1n:>6}, fraud={s1f:.4f}")
        print(f"  Stage 2: n={s2n:>6}, fraud={s2f:.4f}  (Δn={s2n-s1n:+}, Δfraud={s2f-s1f:+.4f})")
        print(f"  Stage 3: n={s3n:>6}, fraud={s3f:.4f}  (Δn={s3n-s2n:+}, Δfraud={s3f-s2f:+.4f})")

    return stage3_df
