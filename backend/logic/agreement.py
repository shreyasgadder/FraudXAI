def explanation_agreement(shap_df, gnn_feat_df, top_k=3):
    shap_top = set(shap_df.head(top_k)["Feature"])
    gnn_top = set(gnn_feat_df.head(top_k)["Feature"])
    overlap = shap_top & gnn_top
    ratio = len(overlap) / top_k if top_k > 0 else 0.0

    return {
        "agreement_count": len(overlap),
        "agreement_ratio": ratio,
        "features": list(overlap)
    }
