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
text → TF-IDF (vocab 50k, ngram 1-2)
     → TruncatedSVD-128 [train-only fit]                ┐
                                                         │
                                                         │ 모달리티별
                                                         │ train-only
                                                         │ z-score
                                                         │
text → frozen SBERT 384D (no_grad, 1회 캐시)              │
     → TruncatedSVD-128 [train-only fit]                ┘
                            │
                            ▼ concat (256D)
                            │
                            ▼ joint TruncatedSVD-128 [train-only fit]
                            text_emb (128D)
                            ⊕ numeric context 12D
                            = Node Feature 140D
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

## 7. 라이선스 / 팀

ITDA Team UnivConcat — 2026 ITDA Networking Day 학술제 본선 제출용.
