import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

# UTF-8 인코딩 설정 (Windows 호환성)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import set_seed, load_config, save_object, save_json
from src.utils.metrics import calculate_metrics, find_best_threshold, print_metrics
from src.utils.html_report import create_html_report
from src.models.baseline_mlp import MLP
from src.models.baseline_gcn import GCN
from src.models.baseline_graphsage import GraphSAGE
from src.models.baseline_gat import GAT
from src.models.baseline_cheb import ChebBaseline
from src.models.baseline_tag import TAGBaseline
from src.models.losses import WeightedBCELoss, FocalLoss, AuxiliaryLoss

DEFAULT_SEED = 42
set_seed(DEFAULT_SEED)


def check_and_preprocess(config):
    """필요한 전처리 파일 확인, 없으면 자동으로 전처리 실행"""
    import subprocess

    processed_dir = config["data_processed"]
    required_files = [
        os.path.join(processed_dir, "node_samples.csv"),
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "edge_index_dict.pt"),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if not missing_files:
        print(f"[Preprocess] ✅ 모든 파일 존재 (스킵)")
        return

    print(f"[Preprocess] ❌ 부족한 파일 발견, 전처리 시작...")

    preprocessing_steps = [
        ("load_yelpzip", "YelpZip CSV 로드"),
        ("label_convert", "라벨 변환"),
        ("sampling", "샘플링"),
        ("feature_engineering", "Feature Engineering"),
    ]

    for step_name, step_desc in preprocessing_steps:
        print(f"\n[{step_desc}] 실행 중...")
        cmd = [sys.executable, "-m", f"src.preprocessing.{step_name}"]
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.parent))
        if result.returncode != 0:
            print(f"[ERROR] {step_desc} 실패")
            raise RuntimeError(f"Preprocessing failed at {step_name}")

    print(f"\n[Graph Relations] 엣지 생성 중...")
    cmd = [sys.executable, "-m", "src.graph.build_relations"]
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.parent))
    if result.returncode != 0:
        print(f"[ERROR] Graph relations 생성 실패")
        raise RuntimeError("Graph building failed")

    print(f"\n[Preprocess] ✅ 전처리 완료")


def load_graph_data(config):
    processed_dir = config["data_processed"]

    nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
    features = np.load(os.path.join(processed_dir, "features.npy"))

    edge_index_dict = torch.load(os.path.join(processed_dir, "edge_index_dict.pt"))

    x = torch.FloatTensor(features)
    y = torch.LongTensor(nodes_df["label"].values)

    train_mask = torch.BoolTensor(nodes_df["split"] == "train")
    valid_mask = torch.BoolTensor(nodes_df["split"] == "valid")
    test_mask = torch.BoolTensor(nodes_df["split"] == "test")

    print(f"[Data] 노드 수: {len(x)}")
    print(f"[Data] Feature 차원: {x.shape[1]}")
    print(f"[Data] Train: {train_mask.sum()} | Valid: {valid_mask.sum()} | Test: {test_mask.sum()}")
    print(f"[Data] Label 분포: {np.bincount(y.numpy())}")

    return x, y, edge_index_dict, train_mask, valid_mask, test_mask, nodes_df


def create_model(model_name, input_dim, config):
    if model_name == "cage_carerf_gnn":
        from src.models.cage_carerf_gnn import CAGECareRF_GNN
        m_cfg = config.get("model", {})
        return CAGECareRF_GNN(
            input_dim=input_dim,
            hidden_dim=m_cfg.get("hidden_dim", 128),
            num_layers=m_cfg.get("num_layers", 3),
            dropout=m_cfg.get("dropout", 0.3),
            K=m_cfg.get("K", 3),
            active_relations=m_cfg.get("active_relations", None),
            use_skip=m_cfg.get("use_skip", True),
            use_gating=m_cfg.get("use_gating", True),
            use_aux_loss=m_cfg.get("use_aux_loss", True),
            care_inline=False,
        )
    if model_name == "mlp":
        cfg = config.get("baselines", {}).get("mlp", {})
        return MLP(input_dim, hidden_dim=cfg.get("hidden_dim", 256), dropout=cfg.get("dropout", 0.3))
    elif model_name == "gcn":
        cfg = config.get("baselines", {}).get("gcn", {})
        return GCN(input_dim, hidden_dim=cfg.get("hidden_dim", 128), num_layers=cfg.get("num_layers", 3), dropout=cfg.get("dropout", 0.3))
    elif model_name == "graphsage":
        cfg = config.get("baselines", {}).get("graphsage", {})
        return GraphSAGE(input_dim, hidden_dim=cfg.get("hidden_dim", 128), num_layers=cfg.get("num_layers", 3), dropout=cfg.get("dropout", 0.3))
    elif model_name == "gat":
        cfg = config.get("baselines", {}).get("gat", {})
        return GAT(input_dim, hidden_dim=cfg.get("hidden_dim", 128), num_layers=cfg.get("num_layers", 3), num_heads=cfg.get("num_heads", 8), dropout=cfg.get("dropout", 0.3))
    elif model_name == "cheb":
        cfg = config.get("baselines", {}).get("cheb", {})
        return ChebBaseline(input_dim, hidden_dim=cfg.get("hidden_dim", 128), num_layers=cfg.get("num_layers", 3), dropout=cfg.get("dropout", 0.3), K=cfg.get("K", 3))
    elif model_name == "tag":
        cfg = config.get("baselines", {}).get("tag", {})
        return TAGBaseline(input_dim, hidden_dim=cfg.get("hidden_dim", 128), num_layers=cfg.get("num_layers", 3), dropout=cfg.get("dropout", 0.3), K=cfg.get("K", 3))
    elif model_name.startswith("cage_rf_gnn"):
        import importlib
        try:
            module = importlib.import_module(f"src.models.{model_name}")
            CAGERF_GNN = module.CAGERF_GNN
        except ImportError:
            raise ValueError(f"Cannot load model: {model_name}")

        cfg = config.get("cage_rf", {})
        extra_kwargs = {}

        if "gat" in model_name:
            extra_kwargs["heads"] = cfg.get("heads", 8)
        elif "cheb" in model_name or "tag" in model_name:
            extra_kwargs["K"] = cfg.get("K", 3)
        elif "sg" in model_name:
            extra_kwargs["K"] = cfg.get("K", 2)

        return CAGERF_GNN(
            input_dim,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.3),
            use_gating=cfg.get("use_gating", True),
            use_ensemble=cfg.get("use_ensemble", False),
            selected_relations=cfg.get("selected_relations", None),
            use_skip_connection=cfg.get("use_skip_connection", False),
            use_two_stage=cfg.get("use_two_stage", False),
            **extra_kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def calculate_pos_weight(y, train_mask):
    pos_count = (y[train_mask] == 1).sum().item()
    neg_count = (y[train_mask] == 0).sum().item()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    return torch.tensor(pos_weight)


def create_loss_fn(config, pos_weight, device):
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "weighted_bce")
    aux_weight = loss_cfg.get("aux_weight", 0.0)

    if loss_type == "focal":
        alpha = loss_cfg.get("focal_alpha", 0.25)
        gamma = loss_cfg.get("focal_gamma", 2.0)
        base_loss = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        base_loss = WeightedBCELoss(pos_weight=pos_weight.to(device))

    if aux_weight > 0:
        return AuxiliaryLoss(main_loss_fn=base_loss, aux_weight=aux_weight)
    else:
        return base_loss


def train_epoch(model, x, y, edge_index_dict, train_mask, optimizer, loss_fn, device,
                oversample_ratio=None, hard_mining_ratio=None, hard_mining_weight=None):
    model.train()

    output = model(x, edge_index_dict)

    if isinstance(output, tuple):
        logits, aux_logits = output
    else:
        logits = output
        aux_logits = None

    train_indices = train_mask.nonzero(as_tuple=True)[0]
    loss_indices = train_indices
    hard_indices = None

    if oversample_ratio is not None and oversample_ratio > 1.0:
        try:
            pos_idx = train_indices[y[train_indices] == 1]
            neg_idx = train_indices[y[train_indices] == 0]
            if len(pos_idx) > 0:
                n_oversample = int(len(pos_idx) * oversample_ratio)
                rand_indices = torch.randint(0, len(pos_idx), (n_oversample,), device=device)
                oversampled_pos = pos_idx[rand_indices]
                loss_indices = torch.cat([neg_idx, oversampled_pos])
        except Exception as e:
            print(f"[Warning] Oversampling error: {e}")
            pass

    if hard_mining_ratio is not None and hard_mining_ratio > 0:
        try:
            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                neg_idx = loss_indices[y[loss_indices] == 0]
                hard_neg = torch.tensor([], dtype=torch.long, device=device)
                if len(neg_idx) > 0:
                    neg_scores = probs[neg_idx]
                    k_neg = max(1, int(len(neg_idx) * hard_mining_ratio))
                    hard_neg = neg_idx[neg_scores.topk(min(k_neg, len(neg_idx))).indices]

                pos_idx = loss_indices[y[loss_indices] == 1]
                hard_pos = torch.tensor([], dtype=torch.long, device=device)
                if len(pos_idx) > 0:
                    pos_scores = probs[pos_idx]
                    k_pos = max(1, int(len(pos_idx) * hard_mining_ratio))
                    hard_pos = pos_idx[pos_scores.topk(min(k_pos, len(pos_idx)), largest=False).indices]

                if len(hard_neg) > 0 or len(hard_pos) > 0:
                    hard_indices = torch.cat([hard_neg, hard_pos])
        except Exception as e:
            print(f"[Warning] Hard mining error: {e}")
            hard_indices = None

    if isinstance(loss_fn, AuxiliaryLoss):
        aux_logits_for_loss = {k: v[loss_indices] for k, v in aux_logits.items()} if aux_logits else None
        main_loss = loss_fn(logits[loss_indices], y[loss_indices].to(device), aux_logits_for_loss)

        if hard_indices is not None and len(hard_indices) > 0:
            try:
                aux_logits_for_hard = {k: v[hard_indices] for k, v in aux_logits.items()} if aux_logits else None
                hard_loss = loss_fn(logits[hard_indices], y[hard_indices].to(device), aux_logits_for_hard)
                loss = main_loss + (hard_mining_weight or 0.5) * hard_loss
            except Exception as e:
                print(f"[Warning] Hard mining loss error: {e}")
                loss = main_loss
        else:
            loss = main_loss
    else:
        loss = loss_fn(logits[loss_indices], y[loss_indices].to(device))

        if hard_indices is not None and len(hard_indices) > 0:
            try:
                hard_loss = loss_fn(logits[hard_indices], y[hard_indices].to(device))
                loss = loss + (hard_mining_weight or 0.5) * hard_loss
            except Exception as e:
                print(f"[Warning] Hard mining loss error: {e}")
                pass

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, x, y, edge_index_dict, mask, device):
    model.eval()

    output = model(x, edge_index_dict)

    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    logits = logits.detach().cpu().numpy()

    y_score = 1 / (1 + np.exp(-logits))
    y_pred = (y_score >= 0.5).astype(int)

    mask_np = mask.cpu().numpy()
    y_true = y.cpu().numpy()[mask_np]

    metrics = calculate_metrics(y_true, y_score[mask_np], y_pred[mask_np])

    return metrics, y_score, y_pred


def train(model_name, config_path, seed=DEFAULT_SEED):
    config = load_config(config_path)

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA not available, falling back to CPU")

    print(f"[Train] 모델: {model_name.upper()}")
    print(f"[Device] {device}")
    print(f"[Seed]   {seed}")

    check_and_preprocess(config)

    x, y, edge_index_dict, train_mask, valid_mask, test_mask, nodes_df = load_graph_data(config)

    x = x.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)
    test_mask = test_mask.to(device)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    # Offline CARE filter (feature cosine top-k, label-free).
    care_cfg = config.get("care_filter", {})
    if care_cfg.get("enabled", False) and care_cfg.get("apply", "offline") == "offline":
        from src.filtering.care_neighbor_filter import filter_edge_index_dict
        top_k = care_cfg.get("top_k", 10)
        min_sim = care_cfg.get("min_sim", None)
        log_path = os.path.join(config.get("output_dir", "outputs"), "metrics", "filter_log.json")
        print(f"\n[CARE] Applying offline neighbor filter (top_k={top_k}, min_sim={min_sim})...")
        edge_index_dict, _ = filter_edge_index_dict(
            x, edge_index_dict, top_k_per_relation=top_k,
            min_sim_per_relation={k: min_sim for k in edge_index_dict} if min_sim is not None else None,
            log_path=log_path,
        )

    model = create_model(model_name, x.shape[1], config)
    model = model.to(device)

    pos_weight = calculate_pos_weight(y, train_mask)
    loss_fn = create_loss_fn(config, pos_weight, device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    print(f"\n[Training] 시작 (epochs={config['training']['num_epochs']})")

    best_valid_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    training_cfg = config.get("training", {})
    oversample_ratio = training_cfg.get("oversample_ratio", None)
    hard_mining_ratio = training_cfg.get("hard_mining_ratio", None)
    hard_mining_weight = training_cfg.get("hard_mining_weight", 0.5)

    try:
        oversample_ratio = float(oversample_ratio) if oversample_ratio else None
    except (ValueError, TypeError):
        oversample_ratio = None

    try:
        hard_mining_ratio = float(hard_mining_ratio) if hard_mining_ratio else None
    except (ValueError, TypeError):
        hard_mining_ratio = None

    try:
        hard_mining_weight = float(hard_mining_weight)
    except (ValueError, TypeError):
        hard_mining_weight = 0.5

    for epoch in range(config["training"]["num_epochs"]):
        loss = train_epoch(
            model, x, y, edge_index_dict, train_mask, optimizer, loss_fn, device,
            oversample_ratio=oversample_ratio,
            hard_mining_ratio=hard_mining_ratio,
            hard_mining_weight=hard_mining_weight
        )

        if (epoch + 1) % config["training"]["validation_interval"] == 0:
            valid_metrics, _, _ = evaluate(model, x, y, edge_index_dict, valid_mask, device)

            if valid_metrics["macro_f1"] > best_valid_f1:
                best_valid_f1 = valid_metrics["macro_f1"]
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(
                    f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                    f"Valid F1: {valid_metrics['macro_f1']:.4f} | PR-AUC: {valid_metrics['pr_auc']:.4f} ✓"
                )
            else:
                patience_counter += 1
                print(
                    f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                    f"Valid F1: {valid_metrics['macro_f1']:.4f} | PR-AUC: {valid_metrics['pr_auc']:.4f}"
                )

            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n[Evaluation]")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    valid_metrics, valid_scores, valid_preds = evaluate(
        model, x, y, edge_index_dict, valid_mask, device
    )
    test_metrics, test_scores, test_preds = evaluate(
        model, x, y, edge_index_dict, test_mask, device
    )

    print("\n[Valid Set]")
    print_metrics(valid_metrics, "Validation Metrics")

    evaluation_cfg = config.get("evaluation", {})
    threshold_metric = evaluation_cfg.get("threshold_type", "macro_f1")

    valid_mask_np = valid_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()

    best_threshold, best_threshold_score = find_best_threshold(
        y.cpu().numpy()[valid_mask_np],
        valid_scores[valid_mask_np],
        metric=threshold_metric
    )

    print(f"\n[Threshold Optimization]")
    print(f"  Metric: {threshold_metric}")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Score: {best_threshold_score:.4f}")

    test_preds_thresholded = (test_scores[test_mask_np] >= best_threshold).astype(int)
    test_metrics_thresholded = calculate_metrics(
        y.cpu().numpy()[test_mask_np], test_scores[test_mask_np], test_preds_thresholded
    )

    print("\n[Test Set]")
    print_metrics(test_metrics_thresholded, "Test Metrics (with best threshold)")

    # 폴더 구조 결정
    if model_name.startswith("cage_rf_gnn_"):
        # GNN 벤치마크: outputs/benchmark/{MODEL_TYPE}/
        gnn_type = model_name.split("_")[-1].upper()  # cage_rf_gnn_gat → GAT
        output_dir = os.path.join("outputs", "benchmark", gnn_type)
        version_suffix = config.get("version", "v7_ensemble")
        version = f"cage_rf_gnn_{gnn_type.lower()}_{version_suffix}"
    elif model_name == "cage_rf_gnn":
        # 기본 CAGE-RF: outputs/cage_rf_gnn/
        output_dir = os.path.join("outputs", "cage_rf_gnn")
        version = config.get("version", "v2")
    elif model_name == "cage_carerf_gnn":
        # CAGE-CareRF (FINAL + ablations): 같은 모델·다른 config가 서로
        # 덮어쓰지 않도록 config의 version 키를 파일명에 합친다.
        output_dir = os.path.join("outputs", "cage_rf_gnn")
        version_suffix = config.get("version", "cage_carerf_v1")
        version = f"{model_name}_{version_suffix}"
    else:
        # Baseline 모델들: outputs/cage_rf_gnn/
        output_dir = os.path.join("outputs", "cage_rf_gnn")
        version = model_name

    os.makedirs(output_dir, exist_ok=True)

    seed_suffix = f"_seed{seed}"
    version_with_seed = f"{version}{seed_suffix}"

    model_path = os.path.join(output_dir, f"best_model_{model_name}{seed_suffix}.pt")
    torch.save(best_model_state, model_path)
    print(f"\n[Save] {model_path}")

    # LGBM stacking 용 valid/test 확률 저장 (probs_*_seed{N}.npy).
    # lgbm_stacking 모듈이 --gnn-probs-valid/test 로 이 파일을 읽어 blend 수행.
    probs_valid_path = os.path.join(output_dir, f"probs_valid{seed_suffix}.npy")
    probs_test_path = os.path.join(output_dir, f"probs_test{seed_suffix}.npy")
    np.save(probs_valid_path, valid_scores[valid_mask_np])
    np.save(probs_test_path, test_scores[test_mask_np])
    print(f"[Save] {probs_valid_path}")
    print(f"[Save] {probs_test_path}")

    metrics_dict = {
        "model": model_name,
        "seed": seed,
        "best_threshold": best_threshold,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics_thresholded,
    }

    metrics_path = os.path.join(output_dir, f"metrics_{version_with_seed}.json")
    save_json(metrics_dict, metrics_path)
    print(f"[Save] {metrics_path}")

    report_path = os.path.join(output_dir, f"report_{version_with_seed}.html")
    create_html_report(model_name, metrics_dict, report_path)
    print(f"[Save] {report_path}")

    return test_metrics_thresholded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cage_rf_gnn",
                        choices=[
                            "mlp", "gcn", "graphsage", "gat", "cheb", "tag",
                            "cage_rf_gnn",
                            "cage_rf_gnn_sage", "cage_rf_gnn_gat", "cage_rf_gnn_gcn",
                            "cage_rf_gnn_graphconv", "cage_rf_gnn_cheb", "cage_rf_gnn_tag",
                            "cage_rf_gnn_sg",
                            "cage_carerf_gnn",
                        ])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing and use existing data")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip graph building and use existing edge_index_dict")
    args = parser.parse_args()

    metrics = train(args.model, args.config, seed=args.seed)
