import torch

def build_explanation_scorecard(
    threshold_dict: dict,
    anchor_idx: int,
    anchor_prob: float,
    fidelity_df,
    gnn_fidelity,
    shap_sparsity,
    edge_sparsity,
    shap_stability,
    gnn_stability,
    # agreement,
    explanation
):
    # Edge concentration (UI justification)
    all_edge_masks = torch.cat([
        m for m in explanation.edge_mask_dict.values()
        if m is not None
    ])

    if all_edge_masks.numel() > 0:
        top_5_weight = torch.topk(
            all_edge_masks,
            k=min(5, all_edge_masks.numel())
        ).values.sum().item()
        total_weight = all_edge_masks.sum().item()
        edge_concentration = round(top_5_weight / total_weight, 4)
    else:
        edge_concentration = 0.0

    def risk_level(prob):
        if prob >= threshold_dict["high"]:
            return "High"
        elif prob >= threshold_dict["medium"]:
            return "Medium"
        else:
            return "Low"

    # def agreement_strength(ratio: float) -> str:
    #     if ratio >= 0.66:
    #         return "STRONG"
    #     elif ratio >= 0.33:
    #         return "PARTIAL"
    #     else:
    #         return "WEAK"

    # agreement_ratio = agreement["agreement_ratio"]
    decision_threshold=threshold_dict["medium"]
    return {
        "metadata": {
            "transaction_id": int(anchor_idx),
            "fraud_probability": float(anchor_prob),
            "risk_level": risk_level(anchor_prob),
            "decision_delta": float(anchor_prob - decision_threshold)
        },
        "tabular_metrics": {
            "faithfulness": fidelity_df.to_dict(orient="records"),
            "sparsity": shap_sparsity,
            "stability": shap_stability,
            "monotonicity": bool(fidelity_df["Drop"].is_monotonic_increasing)
        },
        "structural_metrics": {
            "faithfulness_edges": gnn_fidelity,
            "sparsity_gnn": edge_sparsity,
            "gnn_stability": gnn_stability,
            "edge_concentration": edge_concentration
        },
        # "consensus": {
        #     "agreement_strength": agreement_strength(agreement_ratio),
        #     "agreement_ratio": agreement_ratio,
        #     "agreement_count": agreement["agreement_count"],
        #     "features": agreement["features"]
        # }
    }
