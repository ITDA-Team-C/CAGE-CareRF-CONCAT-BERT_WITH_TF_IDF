# CAGE-CareRF GNN — YelpZip 사기 리뷰 탐지 · **TF-IDF ⊕ SBERT concat 텍스트 인코더 변형**

> **CAGE-RF + CARE** — Camouflage-Aware Gated Edge Relation-Fusion GNN
> ITDA Networking Day 2026 예선 코드 (Team C)
> **FINAL backbone**: CAGE-RF + CARE — Test PR-AUC **0.4447 ± 0.0061** (TF-IDF 단독 인코더 기준, 15모델 중 1위)

> ### 🔖 이 repo의 정체성 — 인코더 fusion 변형 (concat **후**)
> frozen-SBERT 단독이 TF-IDF 대비 PR-AUC에서 손해(주력 모델 −0.0135)였던 한계를,
> **두 인코더를 합쳐** 보완하는 변형. `TEXT_ENCODER` 환경변수로 선택:
> - **`concat`** (본 repo의 주제) — TF-IDF→SVD128 ⊕ frozen-SBERT→SVD128, 모달리티별 train-only z-score 후 concat(256) → **공동 train-only SVD 128**
> - `sbert` — frozen SBERT → SVD 128 (concat 전 변형, 별도 repo `CARE-RF-GNN_With_BERT`)
> - `tfidf` — 기존 TF-IDF → SVD 128 (원 FINAL 인코더)
>
> 셋 다 최종 128차원으로 끝나 downstream(그래프/CARE/모델)이 **byte-for-byte 동일** → 인코더만의 효과를 깨끗이 비교.
>
> **설계 의도**: TF-IDF의 도메인 변별 토큰(lexical 정밀도) + SBERT의 의미·동의어 표현을 결합해
> 각 인코더 단독의 약점을 상호 보완. 공동 SVD로 차원을 128로 되돌려 기존 결과들과 apples-to-apples 유지.

YelpZip 리뷰를 노드로 두고 6개 relation(기본 3 + 커스텀 3)으로 그래프를 구성한 뒤, CARE-GNN의 camouflage-resistant neighbor filtering + Skip GNN branch + Gated Relation Fusion + Auxiliary branch loss를 결합한 다중 관계 GNN.

**최종 보고서**: [`REAL_FINAL_YELPZIP.md`](REAL_FINAL_YELPZIP.md) — 규정 8섹션 양식.

---

## 0. 핵심 결과 한눈에

| 지표 | 값 (mean ± std, 5 seeds) | 순위 (15모델 중) |
|---|:---:|:---:|
| **Test PR-AUC** | **0.4447 ± 0.0061** | **1위 (std 최소)** |
| **Test Macro F1** | 0.6647 ± 0.0066 | 상위 동률 그룹 |
| **Test G-Mean** | 0.6049 ± 0.0405 | Top 5 |
| **Test ROC-AUC** | **0.8250 ± 0.0021** | **1위** |

- 노드 수: **47,125** (규정 [10k, 50k] 충족, 결정론적 이분 탐색 샘플링)
- 분할: 64/16/20 stratified, `random_state=42`, fraud_ratio 11.16% 유지
- 평가 프로토콜: **15 모델 × 5 seeds = 75회 학습** (학회 표준 multi-seed)
- Cross-dataset 검증: Amazon 35회 + YelpChi 35회 = 총 **145회 학습**

---

## 1. 환경

| 항목 | 값 |
|---|---|
| **Python** | 3.11+ 권장 (개발/검증 3.13.12) |
| **OS** | Linux / macOS / Windows |
| **GPU** | 강력 추천 (CPU에서는 75회 학습이 매우 오래 걸림) |
| 검증 환경 | Linux + CUDA 12.x |

---

## 2. 빠른 시작

```bash
# 1) Clone & venv
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE
python -m venv .venv && source .venv/bin/activate

# 2) 의존성
pip install -r requirements.txt

# 3) 데이터 배치 (repo에 미포함, ~410MB)
mkdir -p data/raw
# yelp_zip.csv 를 data/raw/ 에 둠
# 기대 컬럼: user_id, prod_id, rating, label, date, text, tag

# 4) 전처리 + 그래프 빌드 (1회)
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert       # -1 → 1, 1 → 0
python -m src.preprocessing.sampling            # 결정론적 이분 탐색 임계값
TEXT_ENCODER=concat python -m src.preprocessing.feature_engineering  # TF-IDF ⊕ SBERT (sbert/tfidf 도 가능)
python -m src.graph.build_relations             # 6개 relation 빌드 (위 인코더 산출물 기반)
python -m src.graph.relation_quality            # fraud_edge_lift 산출

# 5) 학습 — FINAL 단일 실행
python -m src.training.train --model cage_rf_gnn_cheb \
    --config configs/cage_rf_skip_care.yaml --seed 42

# 6) 학습 — 15 × 5 = 75회 (학회 표준, 권장)
python 5x_run_all_models.py
```

---

## 2-b. concat 인코더 실행 (이 repo 핵심)

```bash
# TF-IDF ⊕ SBERT concat — SBERT 1회 인코딩 후 data/interim 캐시, 이후 재사용
TEXT_ENCODER=concat python -m src.preprocessing.feature_engineering
# 인코더 바꾼 뒤 relation 재빌드 필수 (텍스트 임베딩 [:, :128] 기반)
python -m src.graph.build_relations && python -m src.graph.relation_quality
# 검증 — text_encoder=concat 떠야 정상
python -c "import json,sys;m=json.load(open('data/processed/feature_meta.json'));print(m.get('text_encoder'), m['num_features'])"
# 15모델 × 5 seeds
python 5x_run_all_models.py
```

### concat 파이프라인 (leakage-safe)

```text
TF-IDF(vocab) ──train-only SVD──▶ 128 ──┐
                                         ├─ train-only z-score ─▶ concat (256) ──train-only SVD──▶ 128 ─▶ (+ numeric 12) = 140D
frozen SBERT(384) ─train-only SVD─▶ 128 ─┘
```

- 모든 fit(SVD·StandardScaler·TF-IDF)은 `split=='train'` 행에서만, SBERT는 frozen(미세조정 X) → leakage-safe.
- 모달리티별 z-score 후 합쳐서 공동 SVD → TF-IDF의 큰 특이값 스케일이 SBERT 채널을 압도하지 않도록 함.
- 최종 텍스트 차원 128 유지 → `sbert`/`tfidf` 단독 실행과 모델·그래프 입력이 동일 (apples-to-apples).

### 인코더별 비교 (실측은 본 repo 75회 실행 후 갱신)

| 인코더 | 동기 | 비고 |
|---|---|---|
| `tfidf` | 도메인 변별 토큰 보존 | 원 FINAL, PR-AUC 최고 기준선 |
| `sbert` | 의미·동의어 표현 | 단독 시 PR-AUC −0.0044(평균)~−0.0135(주력) — 별도 repo 참조 |
| **`concat`** | **둘의 약점 상호 보완** | **본 repo — 결과 산출 후 이 표 갱신 예정** |

> concat은 SBERT 단독의 "도메인 토큰 희석"을 TF-IDF lexical anchor로 메우고, TF-IDF의 "의미 무지"를 SBERT로 보완하는 것이 목표. frozen 한계(도메인 미적응) 자체는 남으므로, 추가 개선은 학습 가능한 projection / SBERT fine-tune이 다음 수순.

---

## 3. 15개 모델 학습 명령

> `w/o Custom Relations` (`configs/ablation_no_custom.yaml`)는 커스텀 relation이 0개이므로 대회 규정("기본 ≥ 1 + 커스텀 ≥ 1") 위반. 본 보고에서 제외하여 15개 모델로 운영.
> Lean-6과 `w/o Gating` ablation은 알고리즘적으로 동일하여 후자로 통합. CAGE-RF Skip (v8)과 `w/o CARE` ablation도 동일 플래그라 `CAGE-RF Skip (w/o CARE)` 단일 라벨로 통합. 빈 자리에 ChebConv·TAGConv baseline을 추가.

실행 위치: repo 루트. 모든 산출물은 자동으로 다른 파일(`_seed{N}` 접미사 포함)에 저장됨.

### A. Baseline 6종 (edge = union of 6 relations)

```bash
python -m src.training.train --model mlp        --config configs/default.yaml --seed 42
python -m src.training.train --model gcn        --config configs/default.yaml --seed 42
python -m src.training.train --model gat        --config configs/default.yaml --seed 42
python -m src.training.train --model graphsage  --config configs/default.yaml --seed 42
python -m src.training.train --model cheb       --config configs/default.yaml --seed 42
python -m src.training.train --model tag        --config configs/default.yaml --seed 42
```

### B. CAGE-RF 계열 4종 (multi-relation GNN, 통합 구현)

```bash
python -m src.training.train --model cage_rf_gnn_cheb --config configs/default.yaml             --seed 42  # Base
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip.yaml             --seed 42  # Skip (= w/o CARE)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage.yaml         --seed 42  # Refine (v9)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/cage_rf_skip_care.yaml   --seed 42  # FINAL: + CARE
```

### C. CAGE-CareRF Lean 변종 2종

```bash
# Lean-4: basic 3 + R-Burst-R
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean.yaml --seed 42
# Lean-5: basic 3 + R-Burst-R + R-SemSim-R
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean_5.yaml --seed 42
# (Lean-6은 w/o Gating ablation과 동일하므로 후자로 통합)
```

### D. Ablation 3종

```bash
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_skip.yaml    --seed 42
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_gating.yaml  --seed 42
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_aux.yaml     --seed 42
# (참고) ablation_no_custom — 대회 규정 부적합으로 보고서에서 제외
# (참고) ablation_no_care — CAGE-RF Skip (w/o CARE)와 동일하므로 통합
```

### 일괄 실행 — Multi-seed × 5 (`5x_run_all_models.py`) **권장**

```bash
python 5x_run_all_models.py                                   # 15 × 5 = 75회
python 5x_run_all_models.py --seeds 42 123 2024 7 1234        # 커스텀 시드 (기본값과 동일)
python 5x_run_all_models.py --only baselines                  # baseline 6종만
python 5x_run_all_models.py --only cage_rf                    # CAGE-RF 4종만
python 5x_run_all_models.py --continue-on-error
python 5x_run_all_models.py --dry-run
```

실행 후 자동 집계: `outputs/multi_seed_summary.json` — 모델별 `pr_auc_mean ± std`, `macro_f1_mean ± std`, **`g_mean_mean ± std`** 저장.

---

## 4. 디렉토리 구조

```text
.
├── src/
│   ├── preprocessing/   load_yelpzip / label_convert / sampling / feature_engineering
│   ├── graph/           build_{rur,rtr,rsr,burst,semsim,behavior,relations,relation_quality}
│   ├── filtering/       care_neighbor_filter
│   ├── models/          baseline_{mlp,gcn,gat,graphsage,cheb,tag} / cage_rf_gnn_cheb /
│   │                    skip_cheb_branch / gated_relation_fusion / cage_carerf_gnn / losses
│   ├── training/        train / evaluate / threshold
│   └── utils/           seed / io / metrics (PR-AUC, Macro-F1, G-Mean, ROC-AUC, ...)
├── configs/             default, v8_skip, v9_twostage, cage_rf_skip_care,
│                        cage_carerf{,_lean,_lean_5}, ablation_no_{skip,gating,aux,custom}
├── amazon/              참고 데이터셋: Amazon (CARE-GNN .mat) — 7모델 × 5 seeds
├── yelchi/              참고 데이터셋: YelpChi (CARE-GNN .mat) — 7모델 × 5 seeds
├── docs/                01_code_structure / 02_training_pipeline / 03_model_architecture /
│                        04_setup_and_run
├── REAL_FINAL_YELPZIP.md     최종 보고서 (규정 8섹션 양식)
├── RESULTS.md                3 데이터셋 × 5 seeds 통합 결과 표
├── run_all_models.py         YelpZip 단일 seed launcher
├── 5x_run_all_models.py      YelpZip 15 × 5 = 75회 launcher
├── run_all_amazon.py         Amazon 7모델 단일 seed launcher
├── 5x_run_all_amazon.py      Amazon 7 × 5 = 35회 launcher
├── run_all_yelchi.py         YelpChi 7모델 단일 seed launcher
└── 5x_run_all_yelchi.py      YelpChi 7 × 5 = 35회 launcher
```

`.gitignore`로 제외: `data/`, `outputs/`, `CARE-GNN/`, `PC-GNN/`, `DGFraud/`, `*.md` (단 `README.md` 예외).

---

## 5. FINAL 모델 개요 — CAGE-RF + CARE

```text
Reviews (N=47,125, F=140 = [concat: TF-IDF⊕SBERT]→SVD 128 + numeric 12)
    │
    ▼   [offline] CARE neighbor filter (feature cosine top-k, label-free)
6 relation graphs (basic R-U-R / R-T-R / R-S-R + 커스텀 R-Burst-R / R-SemSim-R / R-Behavior-R)
    │
    ▼
ChebConv K=3 branch × 6 (per relation, hidden=128, num_layers=3, residual Skip)
    │
    ▼   stack → (N, 6, 128)
Gated Relation Fusion (per-node softmax α over 6 branches)
    │
    ▼   (N, 128)
Projection → Main Classifier            → main_logit
            + 6 × Auxiliary heads        → aux_logits per relation

Loss = FocalLoss(α=0.75, γ=2.0) + 0.3 × mean_r BCE(aux_r, y)
threshold @ valid PR-curve F1-max → Test set 1회 평가
```

| 구성 | 값 |
|---|---|
| 클래스 | `src/models/cage_rf_gnn_cheb.py` (Skip + Gating + Aux 통합) + `src/filtering/care_neighbor_filter.py` (offline) |
| Config | `configs/cage_rf_skip_care.yaml` |
| 사용 relation | 6개 모두 |
| 핵심 모듈 | Skip + Gated Fusion + Auxiliary Loss + **CARE filter** 모두 ON |

자세한 구조는 [`docs/03_model_architecture.md`](docs/03_model_architecture.md) 참조.

---

## 6. 15모델 최종 성능 (5 seeds 평균)

PR-AUC 내림차순. 출처: `outputs/multi_seed_summary.json`.

| Rank | Model | PR-AUC | Macro F1 | G-Mean | ROC-AUC |
|:----:|-------|:------:|:--------:|:------:|:-------:|
| 🥇 1 | **CAGE-RF + CARE (FINAL)** | **0.4447 ± 0.0061** | 0.6647 ± 0.0066 | 0.6049 ± 0.0405 | **0.8250 ± 0.0021** |
| 🥈 2 | w/o Gating (ablation) | 0.4334 ± 0.0102 | 0.6622 ± 0.0039 | 0.6063 ± 0.0174 | 0.8201 ± 0.0011 |
| 🥉 3 | w/o Skip (ablation) | 0.4301 ± 0.0163 | **0.6650 ± 0.0048** | **0.6305 ± 0.0238** | 0.8212 ± 0.0020 |
| 4 | CAGE-CareRF Lean-4 | 0.4296 ± 0.0136 | 0.6625 ± 0.0050 | 0.6124 ± 0.0153 | 0.8179 ± 0.0043 |
| 5 | CAGE-CareRF Lean-5 | 0.4255 ± 0.0160 | 0.6577 ± 0.0035 | 0.5838 ± 0.0314 | 0.8174 ± 0.0033 |
| 6 | CAGE-RF Refine (v9) | 0.4249 ± 0.0135 | 0.6625 ± 0.0051 | 0.6103 ± 0.0334 | 0.8224 ± 0.0051 |
| 7 | CAGE-RF Skip (w/o CARE) | 0.4244 ± 0.0133 | 0.6608 ± 0.0036 | 0.6111 ± 0.0288 | 0.8222 ± 0.0050 |
| 8 | CAGE-RF Base | 0.4012 ± 0.0240 | 0.6603 ± 0.0042 | 0.6150 ± 0.0122 | 0.8188 ± 0.0049 |
| 9 | w/o Aux Loss (ablation) | 0.2982 ± 0.0203 | 0.6212 ± 0.0067 | 0.5817 ± 0.0357 | 0.7700 ± 0.0152 |
| 10 | ChebConv (baseline) | 0.2752 ± 0.0055 | 0.6128 ± 0.0036 | 0.6058 ± 0.0128 | 0.7556 ± 0.0035 |
| 11 | TAGConv (baseline) | 0.2749 ± 0.0099 | 0.6098 ± 0.0058 | 0.5928 ± 0.0293 | 0.7503 ± 0.0024 |
| 12 | GraphSAGE (baseline) | 0.2511 ± 0.0143 | 0.6023 ± 0.0077 | 0.5432 ± 0.0235 | 0.7404 ± 0.0081 |
| 13 | MLP (baseline) | 0.2461 ± 0.0047 | 0.5931 ± 0.0047 | 0.5197 ± 0.0364 | 0.7251 ± 0.0030 |
| 14 | GAT (baseline) | 0.2435 ± 0.0064 | 0.6019 ± 0.0033 | 0.5741 ± 0.0269 | 0.7311 ± 0.0055 |
| 15 | GCN (baseline) | 0.2326 ± 0.0039 | 0.5968 ± 0.0018 | 0.5412 ± 0.0251 | 0.7180 ± 0.0019 |

### Ablation 핵심 발견

| 제거 모듈 | ΔPR-AUC vs FINAL | 해석 |
|---|:---:|---|
| w/o Aux Loss | **−0.1465 ⬇⬇⬇** | **압도적** — branch-wise supervision이 핵심 |
| w/o CARE | −0.0203 | 분명한 양의 기여 — FINAL 등극의 결정타 |
| w/o Skip | −0.0146 | variance 내 (Macro F1 / G-Mean에는 +) |
| w/o Gating | −0.0113 | variance 내 (mean fusion과 통계적 동등) |

---

## 7. 예선 규정 준수 요약

- ✅ **Node = Review** 유지 (분류 타깃 = 리뷰; 보조 노드 미사용 — 단일 노드 타입 homogeneous 그래프)
- ✅ YelpZip 원본 → **결정론적 이분 탐색 임계값 샘플링** → 47,125 노드 → 64/16/20 stratified split (`random_state=42`)
- ✅ **무작위 추출 0건** — `df.sample` / `np.random.choice` / `rng.choice` / `np.random.default_rng` 모두 `src/`에 없음 (`grep` 검증 완료)
- ✅ 라벨 변환 `-1→1, 1→0` (`src/preprocessing/label_convert.py:23`)
- ✅ **기본 relation 3개**(R-U-R, R-T-R, R-S-R) + **커스텀 relation 3개**(R-Burst-R, R-SemSim-R, R-Behavior-R)
- ✅ 모든 relation 결정론적 top-k 또는 threshold 적용
- ✅ TF-IDF/SVD/Scaler **train-only fit** (leakage-safe)
- ✅ relation quality 계산 시 **train labels only**
- ✅ threshold는 valid PR-curve에서 결정, **test set은 1회만 평가**
- ✅ **PR-AUC / Macro F1 / G-Mean / ROC-AUC / Precision / Recall / Accuracy** 모두 저장
- ✅ **Multi-seed (5개) 평균** 보고 — 학회 표준

---

## 8. 데이터 핵심 수치

- **원본**: YelpZip 608,458 reviews, 5,044 prod / 260,239 user, fraud_ratio 13.22%
- **샘플** (결정론적 이분 탐색): **47,125 nodes** (Train 30,160 / Valid 7,540 / Test 9,425)
- **샘플 fraud_ratio**: 11.16% (Train/Valid/Test 모두 동일, stratified)
- **임계값**: `head_pu = 6` (상품·사용자 공통), `head_m = 1` (월) → union 자연스럽게 ≤ 50,000
- **Feature**: 140D = 128 (텍스트→SVD; `TEXT_ENCODER`=concat(기본)|sbert|tfidf) + 12 (numeric), train-only fit

### 6 Relations 도메인 가설 vs 실측

| Relation | 도메인 가설 신호 강도 | 도메인 의미 |
|---|---|---|
| R-Burst-R | **최강** | 단기 평판 폭격 (7일/±1점) |
| R-T-R | 강 | 동일 상품 동월 시간 집중 |
| R-U-R | 중 | 동일 사용자 반복 작성 |
| R-S-R | 중하 | 동일 상품 동일 별점 |
| R-SemSim-R | 중하 | TF-IDF→SVD 코사인 top-5 |
| R-Behavior-R | 약 | user 행동 벡터 cosine top-5 |

> 정확한 `fraud_edge_lift` 수치는 `outputs/metrics/relation_quality.json`에서 확인. 보고서 §4-3에서 약신호 relation도 시너지로 가치를 가짐을 입증 (Lean-4/5/6 ablation).

---

## 9. 산출 파일 매핑

`5x_run_all_models.py` 실행 시 각 모델당 5개 파일이 생성됨 (`_seed{42,123,2024,7,1234}`). 총 75개 + 집계 1개.

| # | 모델 | 결과 파일 패턴 |
|---|---|---|
| 1 | MLP | `outputs/cage_rf_gnn/metrics_mlp_seed{N}.json` |
| 2 | GCN | `outputs/cage_rf_gnn/metrics_gcn_seed{N}.json` |
| 3 | GAT | `outputs/cage_rf_gnn/metrics_gat_seed{N}.json` |
| 4 | GraphSAGE | `outputs/cage_rf_gnn/metrics_graphsage_seed{N}.json` |
| 5 | ChebConv | `outputs/benchmark/CHEB/metrics_cheb_seed{N}.json` |
| 6 | TAGConv | `outputs/benchmark/CHEB/metrics_tag_seed{N}.json` |
| 7 | CAGE-RF Base | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_v2_seed{N}.json` |
| 8 | CAGE-RF Skip (w/o CARE) | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_v8_skip_seed{N}.json` |
| 9 | CAGE-RF Refine (v9) | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_v9_twostage_seed{N}.json` |
| 10 | **CAGE-RF + CARE (FINAL)** | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_cage_rf_skip_care_seed{N}.json` |
| 11 | CAGE-CareRF Lean-4 | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_seed{N}.json` |
| 12 | CAGE-CareRF Lean-5 | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_5_seed{N}.json` |
| 13 | w/o Skip | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_skip_seed{N}.json` |
| 14 | w/o Gating | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_gating_seed{N}.json` |
| 15 | w/o Aux Loss | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_aux_seed{N}.json` |
| — | **집계** | `outputs/multi_seed_summary.json` (mean ± std) |

각 JSON에는 `seed`, `best_threshold`, `valid_metrics`, `test_metrics`가 포함되며, `test_metrics` 안에 PR-AUC / Macro F1 / **G-Mean** / ROC-AUC / Precision / Recall / Accuracy / recall_pos / recall_neg가 들어있음.

---

## 10. 문서

- [`REAL_FINAL_YELPZIP.md`](REAL_FINAL_YELPZIP.md) — 최종 보고서 (규정 8섹션 양식)
- [`RESULTS.md`](RESULTS.md) — 3 데이터셋 × 5 seeds 통합 결과 표
- [`docs/01_code_structure.md`](docs/01_code_structure.md) — 디렉토리/파일 역할 / 데이터 흐름
- [`docs/02_training_pipeline.md`](docs/02_training_pipeline.md) — 7단계 파이프라인 / leakage 차단
- [`docs/03_model_architecture.md`](docs/03_model_architecture.md) — CAGE-RF + CARE 구조 / 각 모듈 / Loss
- [`docs/04_setup_and_run.md`](docs/04_setup_and_run.md) — 환경 setup / 의존성 / 재현성 체크리스트

---

## 11. 참고용 데이터셋 (Amazon / YelpChi)

YelpZip이 본 연구의 **메인 데이터셋**. **Amazon · YelpChi**는 동일 backbone GNN의 cross-dataset 일반화 검증용. `amazon/`, `yelchi/` 폴더에 독립 코드.

데이터 형식: **CARE-GNN 표준 `.mat`** (3 relations + 노드/레이블 이미 포함). YelpZip처럼 별도 subgraph sampling 단계가 없고 stratified split만.

### Amazon 결과 (7모델 × 5 seeds = 35회)

| Rank | Model | PR-AUC | Macro F1 | G-Mean |
|:----:|-------|:------:|:--------:|:------:|
| 1 | MLP | **0.8203 ± 0.0242** | **0.9037 ± 0.0026** | **0.8622 ± 0.0043** |
| 2 | CAGE-CareRF | 0.8162 ± 0.0137 | 0.8996 ± 0.0065 | 0.8564 ± 0.0049 |
| 3 | CAGE-CareRF w/o CARE | 0.8117 ± 0.0348 | 0.8944 ± 0.0104 | 0.8410 ± 0.0205 |
| 4 | GraphSAGE | 0.8112 ± 0.0186 | 0.9002 ± 0.0062 | 0.8538 ± 0.0104 |
| 5 | CAGE-CareRF w/o Aux | 0.8043 ± 0.0229 | 0.8905 ± 0.0123 | 0.8538 ± 0.0092 |
| 6 | GCN | 0.2474 ± 0.0134 | 0.6201 ± 0.0091 | 0.5513 ± 0.0434 |
| 7 | GAT | 0.1491 ± 0.0774 | 0.5431 ± 0.0478 | 0.4734 ± 0.1395 |

### YelpChi 결과 (7모델 × 5 seeds = 35회)

| Rank | Model | PR-AUC | Macro F1 | G-Mean |
|:----:|-------|:------:|:--------:|:------:|
| 1 | CAGE-CareRF w/o CARE | **0.7309 ± 0.0171** | **0.8006 ± 0.0074** | 0.8043 ± 0.0194 |
| 2 | CAGE-CareRF w/o Aux | 0.7198 ± 0.0155 | 0.8002 ± 0.0060 | **0.8094 ± 0.0135** |
| 3 | CAGE-CareRF | 0.7114 ± 0.0148 | 0.7969 ± 0.0058 | 0.8017 ± 0.0150 |
| 4 | GraphSAGE | 0.6178 ± 0.0162 | 0.7412 ± 0.0080 | 0.7192 ± 0.0209 |
| 5 | MLP | 0.5080 ± 0.0227 | 0.7012 ± 0.0115 | 0.6742 ± 0.0111 |
| 6 | GCN | 0.2486 ± 0.0095 | 0.5660 ± 0.0052 | 0.4548 ± 0.0197 |
| 7 | GAT | 0.2401 ± 0.0086 | 0.5596 ± 0.0043 | 0.4636 ± 0.0229 |

### Cross-dataset 통합 비교

| 측면 | YelpZip (메인) | Amazon (참고) | YelpChi (참고) |
|---|---|---|---|
| Relations | 6 (basic 3 + custom 3) | 3 (UPU/USU/UVU) | 3 (RUR/RTR/RSR) |
| Node 단위 | Review | User | Review |
| Feature | 140D (TF-IDF + numeric, 직접 생성) | 25D (.mat 제공) | 32D (.mat 제공) |
| 전처리 | 7단계 파이프라인 (sampling 포함) | `.mat` load만 | `.mat` load만 |
| Subgraph 샘플링 | **결정론적 이분 탐색** (무작위 0건) | 해당 없음 | 해당 없음 |
| 학습 모델 수 (보고) | **15개** | **7개** | **7개** |
| Multi-seed 학습 | **75회** (15 × 5) | **35회** (7 × 5) | **35회** (7 × 5) |
| Best PR-AUC | **0.4447** (CAGE-RF + CARE) | 0.8203 (MLP) | 0.7309 (CAGE-CareRF w/o CARE) |

자세한 사항: [`amazon/README.md`](amazon/README.md), [`yelchi/README.md`](yelchi/README.md).

### Cross-dataset 한 번에 (Multi-seed)

```bash
python 5x_run_all_models.py --continue-on-error && \
python 5x_run_all_amazon.py --continue-on-error && \
python 5x_run_all_yelchi.py --continue-on-error
```

총 학습 횟수: YelpZip 75 + Amazon 35 + YelpChi 35 = **145회**.

---

## 12. 라이센스 / 팀

ITDA Team C — 2026 ITDA Networking Day 학술제 예선 제출용.
외부 reference 라이브러리(CARE-GNN, PC-GNN, DGFraud)는 본 repo에 포함되지 않으며, 각 원본 라이센스에 따릅니다.
