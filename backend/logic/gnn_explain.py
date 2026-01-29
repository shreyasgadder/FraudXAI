# ============================================================
# LOCAL GNN EXPLAINER (GraphSAGE-faithful, FIXED)
# ============================================================

import torch
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig


def run_gnn_explainer(
    model,
    data,
    anchor_tx_idx: int,
    anchor_pred: float,
    tx_feature_names,
    account_encoder,
    medium_threshold: float,
    device
):
    """
    Explains the exact k-hop subgraph that influenced the model prediction.
    No pruning. No heuristics. GraphSAGE-faithful.
    """

    # --------------------------------------------------
    # 1. UI tx id == graph tx id (by construction)
    # --------------------------------------------------
    graph_tx_id = anchor_tx_idx

    # --------------------------------------------------
    # 2. STRICT hetero-hop ego graph (MODEL-FAITHFUL)
    # --------------------------------------------------

    # ---- Hop 0: anchor transaction
    tx_hop_0 = {graph_tx_id}

    # ---- Hop 1: accounts connected to anchor tx
    acc_hop_1 = {
        data["transaction"].sender[graph_tx_id].item(),
        data["transaction"].receiver[graph_tx_id].item(),
    }

    # ---- Hop 2: ALL transactions connected to those accounts (BIDIRECTIONAL)
    tx_hop_2 = set()

    # account -> transaction (sends)
    sends_src, sends_dst = data["account", "sends", "transaction"].edge_index
    for acc in acc_hop_1:
        tx_hop_2.update(sends_dst[sends_src == acc].tolist())

    # account <- transaction (receives)
    receives_src, receives_dst = data["transaction", "receives", "account"].edge_index
    for acc in acc_hop_1:
        tx_hop_2.update(receives_src[receives_dst == acc].tolist())

    # ---- Final node sets
    tx_nodes = tx_hop_0 | tx_hop_2
    acc_nodes = acc_hop_1

    tx_subset = torch.tensor(sorted(tx_nodes), device=device)
    acc_subset = torch.tensor(sorted(acc_nodes), device=device)

    # --------------------------------------------------
    # 3. Build induced subgraph (CONNECTED BY DESIGN)
    # --------------------------------------------------
    subgraph = data.subgraph({
        "transaction": tx_subset,
        "account": acc_subset,
    }).to(device)

    # --------------------------------------------------
    # 4. Local index of anchor inside subgraph
    # --------------------------------------------------
    anchor_local_idx = (tx_subset == graph_tx_id).nonzero(as_tuple=True)[0].item()

    # --------------------------------------------------
    # 5. Remap sender / receiver indices (local space)
    # --------------------------------------------------
    local_sender = torch.zeros(subgraph["transaction"].num_nodes, dtype=torch.long)
    local_receiver = torch.zeros_like(local_sender)

    sends = subgraph["account", "sends", "transaction"].edge_index
    receives = subgraph["transaction", "receives", "account"].edge_index

    local_sender[sends[1]] = sends[0]
    local_receiver[receives[0]] = receives[1]

    # --------------------------------------------------
    # 6. Wrapper (IDENTICAL semantics to training)
    # --------------------------------------------------
    class EdgeWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, x_dict, edge_index_dict):
            return self.base(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                sender_indices=local_sender.to(device),
                receiver_indices=local_receiver.to(device),
                tx_raw=x_dict["transaction"],
                apply_edge_dropout=False,
            )

    wrapper = EdgeWrapper(model).to(device)
    wrapper.eval()

    # --------------------------------------------------
    # 7. GNNExplainer (PHENOMENON-LEVEL, CORRECT USAGE)
    # --------------------------------------------------
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(
            epochs=300,
            lr=0.01,
            coeffs={
                "edge_size": 0.005,
                "edge_ent": 0.01,
            },
        ),
        explanation_type="phenomenon",
        edge_mask_type="object",
        node_mask_type="attributes",
        model_config=ModelConfig(
            mode="binary_classification",
            task_level="edge",
            return_type="raw",
        ),
    )

    target = torch.zeros(subgraph["transaction"].num_nodes, device=device)
    target[anchor_local_idx] = int(anchor_pred >= medium_threshold)

    explanation = explainer(
        x=subgraph.x_dict,
        edge_index=subgraph.edge_index_dict,
        index=anchor_local_idx,
        target=target,
    )

    # --------------------------------------------------
    # 8. Feature importance (ANCHOR ONLY)
    # --------------------------------------------------
    node_mask = explanation.node_mask_dict["transaction"]
    feat_importance = node_mask[anchor_local_idx].detach().cpu().numpy()

    feature_df = (
        pd.DataFrame({
            "Feature": tx_feature_names,
            "GNN_Value": feat_importance,
        })
        .sort_values("GNN_Value", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "subgraph": subgraph,
        "wrapper": wrapper,
        "explainer": explainer,
        "explanation": explanation,
        "tx_subset": tx_subset,
        "acc_subset": acc_subset,
        "anchor_local_idx": anchor_local_idx,
        "feature_importance": feature_df,
        "target": target,
    }
