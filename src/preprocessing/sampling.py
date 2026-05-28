import os
import pandas as pd
import numpy as np
from collections import defaultdict
from src.utils import set_seed

set_seed(42)

CONFIG = {
    "interim_dir": "data/interim",
    "processed_dir": "data/processed",
    "input_file": "labeled_data.csv",
    "target_nodes": 25000,
    "min_nodes": 10000,
    "max_nodes": 50000,
    "random_state": 42,
    # 샘플링 전략:
    #   "group_dense"    : fraud-density × activity 점수 상위 그룹 선택 (1-stage)
    #   "cascade"        : group_dense → semantic filter → behavioral reseed (3-stage)
    #   "hybrid_uniform" : 활동량 top-N 만 보는 기존 방식 (fraud-blind)
    "sampling_strategy": "cascade",
    # group_dense (= cascade Stage 1) 파라미터
    "min_group_activity": 3,
    "min_month_activity": 10,
    "activity_cap": 50,
    # cascade Stage 2 (semantic filter) 파라미터
    # Stage 1 의 글로벌 fraud 비율(~25%) 보다 충분히 높/낮은 클러스터만 차별 처리.
    "cascade_s2_high_density": 0.40,    # 클러스터 fraud 밀도 이 이상 → 통째 유지 (high-signal)
    "cascade_s2_low_density": 0.15,     # 이 미만 → low_keep_ratio 만큼 다운샘플 (likely noise)
    "cascade_s2_low_keep_ratio": 0.3,   # low-density 클러스터의 30% 만 유지
    "cascade_s2_noise_keep_ratio": 0.5,
    "cascade_s2_min_cluster_size": 20,
    # cascade Stage 3 (behavioral reseed) 파라미터
    "cascade_s3_burst_window_days": 7,
    "cascade_s3_burst_min_reviews": 5,
    "cascade_s3_normal_recovery_ratio": 0.5,
}


def add_temporal_features(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M")
    df["days_since_epoch"] = (df["date"] - pd.Timestamp("1970-01-01")).dt.days
    return df


def product_user_time_hybrid_sampling(df):
    print("[Sampling] Product-User-Time Hybrid Dense Sampling 시작...")

    df = add_temporal_features(df)

    print(f"  총 데이터: {len(df)}")
    print(f"  상품 수: {df['prod_id'].nunique()}")
    print(f"  사용자 수: {df['user_id'].nunique()}")
    print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")

    prod_counts = df["prod_id"].value_counts()
    print(f"\n  [Product] 상위 10개 상품: {prod_counts.head(10).to_dict()}")

    user_counts = df["user_id"].value_counts()
    print(f"\n  [User] 상위 10명 사용자: {user_counts.head(10).to_dict()}")

    month_counts = df["year_month"].value_counts()
    print(f"\n  [Time] 상위 10개 월: {month_counts.head(10).to_dict()}")

    def _union_size(head_pu, head_m):
        tp = set(prod_counts.head(head_pu).index)
        tu = set(user_counts.head(head_pu).index)
        tm = set(month_counts.head(head_m).index)
        return int(
            (
                df["prod_id"].isin(tp)
                | df["user_id"].isin(tu)
                | df["year_month"].isin(tm)
            ).sum()
        )

    lo, hi = 1, max(2000, len(df) // 100)
    best_head_pu = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        head_m_mid = max(1, mid // 20)
        if _union_size(mid, head_m_mid) <= CONFIG["max_nodes"]:
            best_head_pu = mid
            lo = mid + 1
        else:
            hi = mid - 1

    head_pu = best_head_pu
    head_m = max(1, head_pu // 20)
    top_products = set(prod_counts.head(head_pu).index)
    top_users = set(user_counts.head(head_pu).index)
    top_months = set(month_counts.head(head_m).index)
    print(f"\n  [Threshold-Search] head_pu={head_pu}, head_m={head_m}")
    print(f"  상위 Product 선택: {len(top_products)}")
    print(f"  활동량 많은 User 선택: {len(top_users)}")
    print(f"  리뷰 집중 Month 선택: {len(top_months)}")

    product_mask = df["prod_id"].isin(top_products)
    user_mask = df["user_id"].isin(top_users)
    month_mask = df["year_month"].isin(top_months)

    sampled_df = df[product_mask | user_mask | month_mask].copy()

    print(f"\n  Union (product OR user OR month): {len(sampled_df)}")

    sampled_df = sampled_df.sort_values(
        by=["prod_id", "user_id", "date"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    print(f"\n  최종 샘플: {len(sampled_df)}")
    print(f"  목표: {CONFIG['target_nodes']}, 범위: [{CONFIG['min_nodes']}, {CONFIG['max_nodes']}]")

    assert CONFIG["min_nodes"] <= len(sampled_df) <= CONFIG["max_nodes"], \
        f"샘플 크기 범위 초과: {len(sampled_df)}"

    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df["node_id"] = np.arange(len(sampled_df))

    return sampled_df


def _score_groups_by_fraud_density(df, group_col, min_activity, activity_cap):
    """
    그룹별 점수 = fraud_density × log1p(min(activity, activity_cap))

    - fraud_density: 그룹 내 fraud 비율 (=조직적 어뷰징 신호의 강도)
    - log1p(activity cap): 활동량이 충분해야 density 가 의미 있고,
                          한두 그룹의 거대한 활동량이 점수를 지배하지 않게 cap
    - min_activity 미만 그룹은 통계적 신뢰성이 낮아 후보에서 제외
    """
    g = df.groupby(group_col).agg(n=("label", "size"), n_fraud=("label", "sum"))
    g = g[g["n"] >= min_activity].copy()
    g["density"] = g["n_fraud"] / g["n"]
    g["score"] = g["density"] * np.log1p(np.minimum(g["n"], activity_cap))
    return g.sort_values(["score", "n"], ascending=[False, False])


def group_dense_sampling(df):
    """
    Fraud-density 가 높은 (user / prod / time-window) 그룹을 우선 선택하는 서브그래프 샘플링.

    - 그래프 연결성: 같은 그룹에 속한 리뷰들이 통째로 들어오므로 R-U-R, R-T-R 연결성 자연 보존
    - Fraud 비율: 조직적 어뷰징 그룹을 우선 선택 → 자연스럽게 ↑ (강제 언더샘플링 아님)
    - 대회 주제 "조직적 어뷰징 네트워크 탐지" 와 의미적으로 정합
    """
    print("[Sampling] Group-Dense (fraud-density × activity) Sampling 시작...")

    df = add_temporal_features(df)

    print(f"  총 데이터: {len(df)}")
    print(f"  전체 fraud 비율: {(df['label']==1).mean():.4f}")
    print(f"  상품 수: {df['prod_id'].nunique()}, 사용자 수: {df['user_id'].nunique()}")
    print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")

    users_scored = _score_groups_by_fraud_density(
        df, "user_id",
        min_activity=CONFIG["min_group_activity"],
        activity_cap=CONFIG["activity_cap"],
    )
    prods_scored = _score_groups_by_fraud_density(
        df, "prod_id",
        min_activity=CONFIG["min_group_activity"],
        activity_cap=CONFIG["activity_cap"],
    )
    months_scored = _score_groups_by_fraud_density(
        df, "year_month",
        min_activity=CONFIG["min_month_activity"],
        activity_cap=CONFIG["activity_cap"],
    )

    print(f"\n  [User] 상위 5 (fraud-density × activity):")
    print(users_scored.head(5)[["n", "n_fraud", "density", "score"]])
    print(f"\n  [Product] 상위 5:")
    print(prods_scored.head(5)[["n", "n_fraud", "density", "score"]])
    print(f"\n  [Month] 상위 5:")
    print(months_scored.head(5)[["n", "n_fraud", "density", "score"]])

    users_ranked = users_scored.index.to_numpy()
    prods_ranked = prods_scored.index.to_numpy()
    months_ranked = months_scored.index.to_numpy()

    def _union_size(head_pu, head_m):
        tp = set(prods_ranked[:head_pu])
        tu = set(users_ranked[:head_pu])
        tm = set(months_ranked[:head_m])
        return int((
            df["prod_id"].isin(tp)
            | df["user_id"].isin(tu)
            | df["year_month"].isin(tm)
        ).sum())

    # 기존 hybrid 와 같은 binary search 구조로 max_nodes 이하의 최대 head_pu 탐색
    lo, hi = 1, max(2000, len(df) // 100)
    best_head_pu = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        head_m_mid = max(1, mid // 20)
        if _union_size(mid, head_m_mid) <= CONFIG["max_nodes"]:
            best_head_pu = mid
            lo = mid + 1
        else:
            hi = mid - 1

    head_pu = best_head_pu
    head_m = max(1, head_pu // 20)
    top_users = set(users_ranked[:head_pu])
    top_prods = set(prods_ranked[:head_pu])
    top_months = set(months_ranked[:head_m])

    print(f"\n  [Threshold-Search] head_pu={head_pu}, head_m={head_m}")
    print(f"  상위 fraud-active User 선택: {len(top_users)}")
    print(f"  상위 fraud-active Product 선택: {len(top_prods)}")
    print(f"  상위 fraud-active Month 선택: {len(top_months)}")

    product_mask = df["prod_id"].isin(top_prods)
    user_mask = df["user_id"].isin(top_users)
    month_mask = df["year_month"].isin(top_months)
    sampled_df = df[product_mask | user_mask | month_mask].copy()

    print(f"\n  Union (product OR user OR month): {len(sampled_df)}")

    sampled_df = sampled_df.sort_values(
        by=["prod_id", "user_id", "date"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    fr = (sampled_df["label"] == 1).mean()
    print(f"\n  최종 샘플: {len(sampled_df)}  |  자연스러운 fraud 비율: {fr:.4f}")
    print(f"  목표: {CONFIG['target_nodes']}, 범위: [{CONFIG['min_nodes']}, {CONFIG['max_nodes']}]")

    assert CONFIG["min_nodes"] <= len(sampled_df) <= CONFIG["max_nodes"], \
        f"샘플 크기 범위 초과: {len(sampled_df)}"

    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df["node_id"] = np.arange(len(sampled_df))

    return sampled_df


def train_val_test_split(df):
    print("\n[Split] Train/Valid/Test 분할...")

    from sklearn.model_selection import train_test_split

    train_ratio = 0.64
    valid_ratio = 0.16
    test_ratio = 0.20

    indices = np.arange(len(df))
    labels = df["label"].values

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=CONFIG["random_state"]
    )

    valid_size = valid_ratio / (valid_ratio + test_ratio)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - valid_size,
        stratify=labels[temp_idx],
        random_state=CONFIG["random_state"]
    )

    df["split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[test_idx, "split"] = "test"

    train_mask = df["split"] == "train"
    valid_mask = df["split"] == "valid"
    test_mask = df["split"] == "test"

    print(f"  Train: {train_mask.sum()} ({train_mask.sum() / len(df) * 100:.1f}%)")
    print(f"  Valid: {valid_mask.sum()} ({valid_mask.sum() / len(df) * 100:.1f}%)")
    print(f"  Test:  {test_mask.sum()} ({test_mask.sum() / len(df) * 100:.1f}%)")

    print(f"\n  Train 라벨 분포: {df[train_mask]['label'].value_counts().to_dict()}")
    print(f"  Valid 라벨 분포: {df[valid_mask]['label'].value_counts().to_dict()}")
    print(f"  Test  라벨 분포: {df[test_mask]['label'].value_counts().to_dict()}")

    return df


def save_sampled_data(df):
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    save_path = os.path.join(CONFIG["processed_dir"], "sampled_reviews.csv")
    df.to_csv(save_path, index=False)
    print(f"\n[Save] {save_path}")

    stats_path = os.path.join(CONFIG["processed_dir"], "sampling_stats.txt")
    fraud_ratio = float((df["label"] == 1).mean())
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== Sampling Statistics ===\n\n")
        f.write(f"Total sampled nodes: {len(df)}\n")
        f.write(f"Target: {CONFIG['target_nodes']}\n")
        f.write(f"Range: [{CONFIG['min_nodes']}, {CONFIG['max_nodes']}]\n")
        f.write(f"random_state: {CONFIG['random_state']}\n")
        f.write(f"sampling_strategy: {CONFIG.get('sampling_strategy', 'group_dense')}\n")
        f.write(f"actual_fraud_ratio: {fraud_ratio:.4f}\n\n")

        f.write("Label Distribution:\n")
        f.write(df["label"].value_counts().to_string() + "\n\n")

        f.write("Split Distribution (target 64/16/20):\n")
        f.write(df["split"].value_counts().to_string() + "\n\n")

        f.write("Per-split Fraud Ratio:\n")
        for sp in ["train", "valid", "test"]:
            sub = df[df["split"] == sp]
            if len(sub) > 0:
                f.write(f"  {sp}: n={len(sub)}, fraud={(sub['label']==1).mean():.4f}\n")
        f.write("\n")

        f.write("Unique values:\n")
        f.write(f"Products: {df['prod_id'].nunique()}\n")
        f.write(f"Users: {df['user_id'].nunique()}\n")

    print(f"[Save] {stats_path}")


if __name__ == "__main__":
    input_path = os.path.join(CONFIG["interim_dir"], "labeled_data.csv")
    df = pd.read_csv(input_path)

    strategy = CONFIG.get("sampling_strategy", "group_dense")
    print(f"\n[Strategy] sampling_strategy = '{strategy}'")
    if strategy == "group_dense":
        df = group_dense_sampling(df)
    elif strategy == "cascade":
        from src.sampling.cascade_pipeline import run_cascade
        df = group_dense_sampling(df)  # Stage 1
        df = run_cascade(               # Stage 2 + Stage 3
            df,
            s2_high_density=CONFIG["cascade_s2_high_density"],
            s2_low_density=CONFIG["cascade_s2_low_density"],
            s2_low_keep_ratio=CONFIG["cascade_s2_low_keep_ratio"],
            s2_noise_keep_ratio=CONFIG["cascade_s2_noise_keep_ratio"],
            s2_min_cluster_size=CONFIG["cascade_s2_min_cluster_size"],
            s3_burst_window_days=CONFIG["cascade_s3_burst_window_days"],
            s3_burst_min_reviews=CONFIG["cascade_s3_burst_min_reviews"],
            s3_normal_recovery_ratio=CONFIG["cascade_s3_normal_recovery_ratio"],
            random_state=CONFIG["random_state"],
        )
        # cascade 출력이 max_nodes 를 넘기지는 않는지 가드 (Stage 3 가 노드 복원하므로)
        assert CONFIG["min_nodes"] <= len(df) <= CONFIG["max_nodes"], \
            f"cascade 결과 노드 수 범위 초과: {len(df)}"
    elif strategy == "hybrid_uniform":
        df = product_user_time_hybrid_sampling(df)
    else:
        raise ValueError(f"unknown sampling_strategy: {strategy}")

    df = train_val_test_split(df)
    save_sampled_data(df)

    print("\n[Done] Phase 2-3: Sampling 완료")
