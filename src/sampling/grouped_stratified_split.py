"""
그룹 단위 층화 분할 (Stratified Group Split for Anomaly Detection)

핵심 아이디어:
  - 시간/장비/사용자/세션 같은 '그룹 ID'가 train·test에 쪼개지면 누수 발생
  - 그룹을 통째로 한 쪽 split에만 배치 + 라벨 비율은 split 간 유사하게 유지
  - 시계열인 경우 시간 순서를 보존하는 옵션도 제공

지원 모드:
  - "shuffle"      : StratifiedGroupKFold (sklearn ≥ 1.0) 기반, 셔플 허용
  - "time_ordered" : group 단위로 시간 정렬 후 앞쪽=train, 중간=valid, 뒤=test
                    (라벨 비율이 시간에 따라 변하면 시간 윈도우 내에서 stratify)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import StratifiedGroupKFold


def _ratios_report(df: pd.DataFrame, label_col: str, group_col: str) -> dict:
    return {
        "n_samples": len(df),
        "n_groups": int(df[group_col].nunique()),
        "anomaly_ratio": float((df[label_col] == 1).mean()),
        "anomaly_groups": int(df.loc[df[label_col] == 1, group_col].nunique()),
    }


def _check_no_group_leak(train_df, valid_df, test_df, group_col) -> dict:
    g_tr = set(train_df[group_col])
    g_va = set(valid_df[group_col])
    g_te = set(test_df[group_col])
    return {
        "train_valid_overlap": len(g_tr & g_va),
        "train_test_overlap": len(g_tr & g_te),
        "valid_test_overlap": len(g_va & g_te),
    }


def grouped_stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "user_id",
    test_size: float = 0.15,
    valid_size: float = 0.15,
    mode: str = "shuffle",
    time_col: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    그룹 단위 무누수 + 라벨 층화 분할.

    Parameters
    ----------
    df : 원본 DataFrame. label_col, group_col(필수), time_col(시계열 모드일 때) 포함.
    label_col : 라벨 컬럼 (이상 탐지: 0=정상, 1=이상).
    group_col : 그룹 ID (user_id / device_id / session_id 등).
    mode :
        - "shuffle"      : StratifiedGroupKFold 기반, 라벨 비율 정렬 우선
        - "time_ordered" : 그룹의 최초 발생 시간 순으로 정렬 → 앞=train, 가운데=valid, 뒤=test
    time_col : mode="time_ordered"일 때 필요 (datetime 또는 정렬 가능한 컬럼).
    """
    assert mode in {"shuffle", "time_ordered"}, f"unknown mode: {mode}"
    assert group_col in df.columns, f"{group_col} not in df"
    assert label_col in df.columns, f"{label_col} not in df"
    df = df.reset_index(drop=True).copy()

    if mode == "shuffle":
        train_df, valid_df, test_df = _split_shuffle(
            df, label_col, group_col, test_size, valid_size, random_state
        )
    else:
        assert time_col is not None and time_col in df.columns, "time_ordered 모드는 time_col 필요"
        train_df, valid_df, test_df = _split_time_ordered(
            df, label_col, group_col, time_col, test_size, valid_size
        )

    leak = _check_no_group_leak(train_df, valid_df, test_df, group_col)
    report = {
        "mode": mode,
        "train": _ratios_report(train_df, label_col, group_col),
        "valid": _ratios_report(valid_df, label_col, group_col),
        "test": _ratios_report(test_df, label_col, group_col),
        "group_leak": leak,
    }

    if verbose:
        print(f"[mode] {mode}")
        for split_name in ("train", "valid", "test"):
            r = report[split_name]
            print(f"[{split_name}] n={r['n_samples']:>6}  groups={r['n_groups']:>5}  "
                  f"anomaly={r['anomaly_ratio']:.3f}  anom_groups={r['anomaly_groups']}")
        print(f"[group_leak] {leak}  (모두 0이어야 무누수)")

    return train_df, valid_df, test_df, report


def _split_shuffle(df, label_col, group_col, test_size, valid_size, random_state):
    """StratifiedGroupKFold를 두 번 적용해 train/valid/test."""
    y = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()

    # 1단계: test 분리
    n_splits_outer = max(2, int(round(1 / test_size)))
    sgkf1 = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    trv_idx, test_idx = next(iter(sgkf1.split(df, y, groups)))

    # 2단계: train/valid 분리 (남은 데이터 기준)
    df_trv = df.iloc[trv_idx].reset_index(drop=True)
    y_trv = df_trv[label_col].to_numpy()
    g_trv = df_trv[group_col].to_numpy()

    valid_ratio_within = valid_size / (1.0 - test_size)
    n_splits_inner = max(2, int(round(1 / valid_ratio_within)))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
    tr_rel, va_rel = next(iter(sgkf2.split(df_trv, y_trv, g_trv)))

    train_df = df_trv.iloc[tr_rel].reset_index(drop=True)
    valid_df = df_trv.iloc[va_rel].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, valid_df, test_df


def _split_time_ordered(df, label_col, group_col, time_col, test_size, valid_size):
    """
    그룹별 최초 발생 시간으로 정렬 → 앞=train, 가운데=valid, 뒤=test.
    각 그룹은 통째로 한 split에만 들어감 → 누수 차단 + 미래 누수도 차단.
    """
    group_first_time = df.groupby(group_col)[time_col].min().sort_values()
    ordered_groups = group_first_time.index.to_numpy()

    n_groups = len(ordered_groups)
    n_test = max(1, int(round(n_groups * test_size)))
    n_valid = max(1, int(round(n_groups * valid_size)))
    n_train = n_groups - n_valid - n_test
    assert n_train > 0, "train에 할당될 그룹이 없습니다. 비율을 조정하세요."

    train_groups = set(ordered_groups[:n_train])
    valid_groups = set(ordered_groups[n_train:n_train + n_valid])
    test_groups = set(ordered_groups[n_train + n_valid:])

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    valid_df = df[df[group_col].isin(valid_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    # 시계열 모드에선 라벨 비율이 시간에 따라 변할 수 있음 → 경고만 출력
    ar_tr = (train_df[label_col] == 1).mean()
    ar_te = (test_df[label_col] == 1).mean()
    if abs(ar_tr - ar_te) > 0.05:
        print(f"[warn] train/test 이상 비율 차이 {abs(ar_tr-ar_te):.3f} > 0.05 "
              f"(시간에 따른 분포 변화 의심 — 정상 현상일 수 있음)")

    return train_df, valid_df, test_df


if __name__ == "__main__":
    # 사용 예시
    rng = np.random.default_rng(0)
    N = 3000
    user_ids = rng.integers(0, 200, size=N)
    times = pd.date_range("2026-01-01", periods=N, freq="h")
    y = (rng.random(N) < 0.05).astype(int)
    df = pd.DataFrame({"user_id": user_ids, "ts": times, "label": y, "feat": rng.normal(size=N)})

    print("=== shuffle 모드 ===")
    tr, va, te, rep = grouped_stratified_split(
        df, label_col="label", group_col="user_id",
        test_size=0.15, valid_size=0.15, mode="shuffle",
    )
    print()
    print("=== time_ordered 모드 ===")
    tr, va, te, rep = grouped_stratified_split(
        df, label_col="label", group_col="user_id", time_col="ts",
        test_size=0.15, valid_size=0.15, mode="time_ordered",
    )
