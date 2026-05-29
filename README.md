# CAGE-CareRF GNN · TF-IDF ⊕ SBERT concat 인코더 변형 (encoder ablation 2단계, 최종 채택)

본 repo는 ITDA Networking Day 2026 학술제 본선 연구의 **인코더 ablation 2단계** 코드이며, 5종 인코더 비교 끝에 **공식 최종 인코더로 채택된 concat 변형** 을 단독으로 다룬다. 1단계(frozen-SBERT 단독)에서 관찰된 상보적 trade-off — PR-AUC 손해 vs Macro F1·G-Mean 우세 — 를 TF-IDF lexical 신호와 SBERT semantic 신호의 결합으로 동시에 회수하는 것이 설계 동기다.

---

## 0. 공식 제출 자료 및 단계별 reference

| 항목 | 위치 |
|---|---|
| **메인 코드 (공식 제출)** | https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE |
| **시각화 결과물** | https://github.com/ITDA-Team-C/Streamlit_CAGE-RF-with-CARE |
| encoder ablation 1단계 · frozen-SBERT | https://github.com/ITDA-Team-C/CARE-RF-GNN_With_BERT |
| 본 repo (encoder ablation 2단계 · concat 최종 채택) | (현재 repo) |
| 추가 실험 · Linear projection 변형 | https://github.com/MeDeoDuck/FINAL_FINAL_EXPERIMENT |

---

## 1. 이 repo의 정체성

동일 GNN backbone 위에서 다음 세 가지 인코더를 모두 비교 가능하게 통합한 변형이다.

- **`concat`** (기본값, 본 repo 주제, **공식 최종 채택**) — TF-IDF SVD-128 ⊕ frozen-SBERT SVD-128, 모달리티별 train-only z-score 후 concat(256) → 공동 train-only SVD-128
- **`sbert`** — frozen SBERT → train-only SVD-128 (1단계 변형)
- **`tfidf`** — TF-IDF → train-only SVD-128 (원 FINAL 인코더, baseline)

세 분기 모두 최종 128차원으로 끝나 downstream(그래프 · CARE · 모델 · loss) 코드는 **byte-for-byte 동일**하게 유지된다.

### 설계 의도
TF-IDF의 도메인 변별 토큰(lexical 정밀도)과 SBERT의 의미·동의어 표현(semantic 일반화) 두 신호의 각자 약점을 상호 보완. 공동 SVD로 차원을 128로 되돌려 기존 변형들과 apples-to-apples 비교를 유지한다.

---

## 2. 2단계 실험 결과 — 5 seeds 인코더 비교 (FINAL backbone)

| Rank | 변형 | PR-AUC | Macro F1 | G-Mean | ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | TF-IDF | 0.4447 ± 0.0061 | 0.6647 ± 0.0066 | 0.6049 ± 0.0405 | 0.8250 ± 0.0021 |
| 2 | **concat (본 repo, 채택)** | 0.4439 ± 0.0152 | **0.6670 ± 0.0036** | **0.6131 ± 0.0257** | 0.8230 ± 0.0040 |
| 3 | frozen-SBERT | 0.4313 ± 0.0131 | **0.6686 ± 0.0074** | **0.6179 ± 0.0223** | 0.8159 ± 0.0038 |

**채택 근거**
- PR-AUC 1위 = TF-IDF, Macro F1 1위 = frozen-SBERT — 단일 인코더는 한 지표에 편중
- concat은 PR-AUC 0.4439(TF-IDF와 Δ−0.0008, std 내 통계적 동률) + Macro F1 0.6670(SBERT와 Δ−0.0016, 통계적 동률)
- → 두 핵심 지표 모두에서 1위와 std 내 동률을 동시에 달성한 **유일한 균형형 인코더**
- 사기 탐지력(PR-AUC) 과 양 클래스 균형(Macro F1) 두 측면을 한 인코더로 만족하므로 공식 최종 채택

---

## 3. 환경

| 항목 | 값 |
|---|---|
| Python | 3.11+ |
| GPU | 강력 추천 |
| 핵심 의존성 | PyTorch · PyG · sentence-transformers · scikit-learn · pandas |

```bash
git clone https://github.com/ITDA-Team-C/CAGE-RF-GNN-CONCAT-BERT_WITH_TF_DF.git
cd CAGE-RF-GNN-CONCAT-BERT_WITH_TF_DF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers
```

---

## 4. 빠른 실행 — concat 변형 (공식 최종)

```bash
# 데이터 배치 (~410MB, repo 미포함)
mkdir -p data/raw
# yelp_zip.csv 를 data/raw/ 에 둠

# 전처리 → 그래프 빌드 → 학습 (TEXT_ENCODER=concat 가 기본값)
TEXT_ENCODER=concat python -m src.preprocessing.feature_engineering
python -m src.graph.build_relations
python -m src.graph.relation_quality

# FINAL 단일 학습
python -m src.training.train --model cage_rf_gnn_cheb \
    --config configs/cage_rf_skip_care.yaml --seed 42

# 학회 표준 multi-seed (15 × 5 = 75회)
python 5x_run_all_models.py
```

다른 인코더로 다시 돌리려면 `TEXT_ENCODER=tfidf` 또는 `TEXT_ENCODER=sbert` 설정 후 위 흐름 재실행.

---

## 5. concat 인코더 구현

```text
.
├── src/
│   ├── preprocessing/   load_yelpzip / label_convert / sampling / feature_engineering
│   ├── sampling/        cascade_pipeline / hdbscan_stratified_split / grouped_stratified_split
│   ├── graph/           build_{rur,rtr,rsr,burst,semsim,behavior,relations,relation_quality}
│   ├── filtering/       care_neighbor_filter
│   ├── models/          baseline_{mlp,gcn,gat,graphsage,cheb,tag} / cage_rf_gnn_cheb /
│   │                    skip_cheb_branch / gated_relation_fusion / cage_carerf_gnn /
│   │                    text_projection_wrapper / losses
│   ├── training/        train / evaluate / threshold /
│   │                    lgbm_stacking / stacked_ensemble / aggregate_final
│   └── utils/           seed / io / metrics
├── configs/             default, v8_skip, v9_twostage, cage_rf_skip_care,
│                        cage_carerf{,_lean,_lean_5}, ablation_no_{skip,gating,aux,custom}
├── amazon/              참고 데이터셋: Amazon — 7모델 × 5 seeds
├── yelchi/              참고 데이터셋: YelpChi — 7모델 × 5 seeds
├── docs/                01_code_structure / 02_training_pipeline / 03_model_architecture /
│                        04_setup_and_run
├── run_all_models.py            YelpZip 단일 seed launcher
├── 5x_run_all_models.py         YelpZip 15 × 5 = 75회 launcher
├── run_proj_experiments.py      Linear projection 변형 2 × 5 = 10회 launcher
├── run_all_amazon.py            Amazon 7모델 launcher
└── run_all_yelchi.py            YelpChi 7모델 launcher
```

- SBERT 본체는 **frozen** — 어떤 split에도 fit/finetune 안 함
- TF-IDF · 모든 SVD · 모든 StandardScaler는 `split == 'train'` 행에서만 fit
- 모달리티별 z-score로 TF-IDF의 큰 특이값 스케일이 SBERT 채널을 압도하지 않도록 정렬
- 최종 텍스트 차원 128로 통일 → 다른 변형과 apples-to-apples 비교 보장

---

## 6. 다음 단계

본 repo는 **encoder ablation 2단계** 의 결론이자 공식 최종 인코더 채택 근거 자료다. 추가 실험으로 학습 가능한 Linear projection을 두 변형(`sbert_proj`, `concat_proj`)에 얹어 보았으나, fraud 라벨 부족 환경에서 단순 SVD를 능가하지 못한다는 negative result가 확인되었다(추가 실험 repo 참조).

따라서 공식 제출 시스템은 **CAGE-RF + CARE backbone + concat encoder** 조합으로 확정된다.

---

## 8. 예선 규정 준수 요약

- Node = Review 유지 (분류 타깃 = 리뷰, 단일 노드 타입 homogeneous 그래프)
- YelpZip 원본 → 결정론적 이분 탐색 임계값 샘플링 → 47,125 노드 → 64/16/20 stratified split (`random_state=42`)
- **무작위 추출 0건** (`df.sample` / `np.random.choice` / `rng.choice` / `np.random.default_rng` 모두 `src/` 에 없음)
- 라벨 변환 `-1→1, 1→0` (`src/preprocessing/label_convert.py`)
- 기본 relation 3 + 커스텀 relation 3 = 6 relations, 모두 결정론적 top-k 또는 threshold 적용
- TF-IDF/SVD/Scaler **train-only fit** (leakage-safe), SBERT는 frozen
- relation quality 계산 시 train labels only
- threshold 는 valid PR-curve에서 결정, **test set 은 1회만 평가**
- PR-AUC / Macro F1 / G-Mean / ROC-AUC / Precision / Recall / Accuracy 모두 저장
- 학회 표준 **multi-seed (5개) 평균 ± std** 보고

---

## 9. 도메인 일반화 — Cross-dataset 검증 (참고)

동일 backbone(CAGE-RF + CARE)을 다른 fraud 도메인에 이식한 결과 (5 seeds 평균):

| 데이터셋 | 제안 모델 | 최고 baseline | PR-AUC (제안) | PR-AUC Δ |
|---|---|---|:---:|:---:|
| YelpZip (메인) | CAGE-RF + CARE | ChebConv | 0.4447 | +0.1696 |
| Amazon | CAGE-CareRF | MLP | 0.8162 | -0.0041 |
| YelpChi | CAGE-CareRF | GraphSAGE | 0.7114 | +0.0936 |

큰 재설계 없이 다른 fraud 도메인으로 이식 가능. 자세한 표는 `amazon/` 및 `yelchi/` 폴더 참조.

---

## 10. 본선 확장 — Behavioral Stacking & Meta-Ensemble

본 섹션은 본선 단계에서 추가된 *후속 실험* 으로, FINAL GNN 의 출력을 **트리 기반 학습기 3종 (LightGBM · XGBoost · CatBoost) 의 행동 피처 모델** 과 결합하여 한 단계 더 성능 마진을 짜내는 stacking 파이프라인이다. 위 1~5 섹션의 공식 FINAL 결과 (`PR-AUC 0.4439 ± 0.0152`) 는 변경되지 않는다.

### 10-1. 누수 차단 — 모든 aggregate 는 train 만으로 계산

```python
# src/training/lgbm_stacking.py  (build_features)
train_df = df.loc[train_mask].copy()
user_agg = _user_aggregates(train_df)   # train 만 사용
prod_agg = _prod_aggregates(train_df)   # train 만 사용
# valid/test 행은 user_id / prod_id 로 *train aggregate* 만 조회
# train 에 없던 user/prod 는 NaN → 0 + unseen_in_train 별도 flag
```

YelpZip fraud 탐지에서 흔히 보이는 `user_fraud_rate` / `prod_fraud_rate` 류의 **전역 집계 leakage** 를 구조적으로 차단. 본 파이프라인 결과는 hold-out test 에 대해 재현 가능한 수치임을 보장한다.

### 10-2. 행동 피처 구성 (총 32개)

| 그룹 | 피처 |
|---|---|
| **self** | rating, rating_is_extreme, text_len, n_words, n_exclaim, n_question, caps_ratio, avg_word_len, ttr, dow, month, year |
| **user_agg** (train only) | n_reviews, rating_{mean,std}, rating_extreme_ratio, text_len_{mean,std}, unique_prods, review_per_prod, unseen_in_train |
| **prod_agg** (train only) | n_reviews, rating_{mean,std,skew}, extreme_ratio, unique_users, review_per_user, unseen_in_train |
| **burst** | 같은 prod 의 ±7일 윈도우 안의 리뷰 수 (binary-search O(N log N)) |
| **relative** | rating − user_mean, rating − prod_mean, text_len − user_mean |

### 10-3. 아키텍처 — 2-Level Stacking

```text
                   ┌─────────────────────────┐
                   │  Behavioral Features    │
                   │  (32 cols, train-only)  │
                   └────────────┬────────────┘
                                │
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
   ┌────────┐     ┌──────────┐    ┌──────────┐    ┌────────┐
   │ LGBM   │     │ XGBoost  │    │ CatBoost │    │  GNN   │ ← saved npy
   │ Level1 │     │ Level1   │    │ Level1   │    │ FINAL  │
   └───┬────┘     └────┬─────┘    └────┬─────┘    └───┬────┘
       │               │               │               │
       └───────────────┴────┬──────────┴───────────────┘
                            ▼
                  ┌──────────────────────┐
                  │  Level-2 Meta-Learner│
                  │  Logistic Regression │  ← valid 에서만 학습
                  │   (4 prob inputs)    │
                  └──────────┬───────────┘
                             ▼
                       Test PR-AUC / Macro-F1
```

- **Level 1**: 네 모델 각각 *독립* 학습. LGBM/XGB/Cat 은 같은 32D 행동 피처 사용, GNN 은 이미 학습된 5-seed 결과의 `probs_*_seed{N}.npy` 를 로드.
- **Level 2**: Logistic Regression 메타 학습기가 `[lgbm_v, xgb_v, cat_v, gnn_v]` 4개 확률을 입력으로 받아 *valid 에서만* 학습 (test 한 번도 미접촉). 가중치는 base learner 별 신뢰도를 *피처 별로* 학습하므로 단순 weighted average 보다 정교.
- **Alternate**: `--meta xgb` 옵션으로 얕은 XGBoost (max_depth=3) 메타 학습기 대안 제공.

### 10-4. 실행

```bash
# 의존성 (한 번만)
pip install lightgbm xgboost catboost hdbscan

# (선행) FINAL GNN 5-seed 학습 — probs_*_seed{N}.npy 자동 저장
python -m src.training.train --model cage_rf_gnn_cheb \
    --config configs/cage_rf_skip_care.yaml --seed 42   # 42, 123, 2024, 7, 1234

# 단순 weighted blend (LGBM only + GNN, 5 seeds)
python -m src.training.lgbm_stacking --seed 42 \
    --gnn-probs-valid outputs/benchmark/CHEB/probs_valid_seed42.npy \
    --gnn-probs-test  outputs/benchmark/CHEB/probs_test_seed42.npy

# Level-2 meta-ensemble (LGBM + XGB + Cat + GNN, 5 seeds)
python -m src.training.stacked_ensemble --seed 42 \
    --gnn-probs-valid outputs/benchmark/CHEB/probs_valid_seed42.npy \
    --gnn-probs-test  outputs/benchmark/CHEB/probs_test_seed42.npy

# 5-seed 통합 리포트 (GNN / LGBM-only / Blend / Meta 한 표)
python -m src.training.aggregate_final
```

### 10-5. 보조 — Cascade Sampling Pipeline (`src/sampling/`)

본선 단계 부수 산출물. 기존 *fraud-blind* hybrid sampling 대비 **자연스럽게 fraud 비율을 25%+ 로 끌어올리는** 3-stage 캐스케이드를 제공한다 (`src/preprocessing/sampling.py` CONFIG 의 `sampling_strategy` 로 선택):

1. **`group_dense`** — fraud-density × log-activity 점수 상위 (user / prod / month) 그룹 선택
2. **`cascade`** — (1) → HDBSCAN semantic 필터링 → R-U-R / burst window 기반 normal reseed (그래프 연결성 회복)
3. 보조 분할 도구: `hdbscan_stratified_split` (의미 군집 + 라벨 동시 stratify), `grouped_stratified_split` (StratifiedGroupKFold, shuffle / time_ordered 모드)

공식 FINAL 결과 (`PR-AUC 0.4439`) 는 기존 hybrid sampling 기준이며, 본 캐스케이드는 본선 분석·실험용으로 분리해 둔다.

---

## 11. 라이선스 / 팀

ITDA Team UnivConcat — 2026 ITDA Networking Day 학술제 본선 제출용.
