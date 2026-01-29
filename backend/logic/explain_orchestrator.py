import torch
import pandas as pd
from .fidelity import tabular_fidelity, gnn_fidelity
from .sparsity import compute_sparsity
from .stability import shap_stability, gnn_stability
from .agreement import explanation_agreement
from .scorecard import build_explanation_scorecard
from .bac import compute_enhanced_bac_score, format_bac_for_ui
from .explanation_materializer import (
    materialize_graph_for_ui,
    detect_structural_pattern,
    generate_investigator_narrative
)


def run_full_explanation_pipeline(
    *,
    anchor_tx_idx: int,

    shap_output: dict,
    gnn_output: dict,

    xgb_model,
    shap_explainer,
    tx_xgb_df,
    train_df,
    tx_feature_names,
    account_encoder,

    threshold_dict,
    anchor_prob: float,

    # NEW parameters for enhanced GNN explainer
    graph_context: dict = None,
    tx_amount: float = 0.0,
    tx_hour: int = 0,

    k_list=(1, 3, 5, 10),
):
    # ==================================================
    # 1. MATERIALIZE EXPLANATIONS
    # ==================================================
    shap_feat_df = shap_output["shap_df"]
    
    gnn_feat_df = gnn_output["feature_importance"]

    merged_df = pd.merge(shap_feat_df, gnn_feat_df, on="Feature", how="outer")
    merged_df = merged_df[merged_df["Feature"] != "xgb_prob"]
    merged_df["Importance"] = (merged_df["SHAP_Value"].fillna(0) * 0.2) \
        + (merged_df["GNN_Value"].fillna(0) * 0.8)
    
    feat_df = merged_df[["Feature", "Importance"]]
    feat_df = feat_df.sort_values(by="Importance", ascending=False)

    print("✅ Explanation output materails Loaded")

    # ==================================================
    # 2. FAITHFULNESS
    # ==================================================
    fidelity_df = tabular_fidelity(
        model_predict_fn=lambda X: xgb_model.predict_proba(X)[:, 1],  
        X_anchor=shap_output["X_anchor"],
        shap_df=shap_output["shap_df"],
        train_df=train_df,
        k_list=list(k_list)
    )

    gnn_fid = gnn_fidelity(
        wrapper=gnn_output["wrapper"],
        subgraph=gnn_output["subgraph"],
        explanation=gnn_output["explanation"],
        anchor_local_idx=gnn_output["anchor_local_idx"],
        top_k=15
    )

    print("✅ Faithfulness computed:")
    print("   tabular fidelity: ", fidelity_df)
    print("   gnn fidelity: ", gnn_fid)

    # ==================================================
    # 3. SPARSITY
    # ==================================================
    shap_sparsity = compute_sparsity(
        shap_output["shap_df"]["SHAP_Value"].values
    )

    edge_importances = torch.cat([
        m for m in gnn_output["explanation"].edge_mask_dict.values()
        if m is not None
    ])
    edge_sparsity = compute_sparsity(edge_importances.cpu().numpy())

    print("✅ Sparsity computed:")
    print("   shap sparsity: ", shap_sparsity)
    print("   edge sparsity: ", edge_sparsity)

    # ==================================================
    # 4. STABILITY 
    # ==================================================
    shap_stab = shap_stability(
        shap_values=shap_output["shap_values"],
        shap_explainer=shap_explainer,
        X_anchor=shap_output["X_anchor"],
        tx_xgb_df=tx_xgb_df
    )

    gnn_stab = gnn_stability(
        explainer=gnn_output["explainer"],
        subgraph=gnn_output["subgraph"],
        anchor_local_idx=gnn_output["anchor_local_idx"],
        target=gnn_output["target"]
    )

    print("✅ Stability computed:")
    print("   shap stability: ", shap_stab)
    print("   gnn stability: ", gnn_stab)

    # ==================================================
    # 5. AGREEMENT
    # ==================================================
    # agreement = explanation_agreement(
    #     shap_feat_df,
    #     gnn_feat_df,
    #     top_k=3
    # )

    # print("✅ Agreement computed:")
    # print("   agreement count: ", agreement["agreement_count"])
    # print("   agreement ratio: ", agreement["agreement_ratio"])
    # print("   features: ", list(agreement["features"]))

    # ==================================================
    # 6. SCORECARD
    # ==================================================
    scorecard = build_explanation_scorecard(
        threshold_dict=threshold_dict,
        anchor_idx=anchor_tx_idx,
        anchor_prob=anchor_prob,
        fidelity_df=fidelity_df,
        gnn_fidelity=gnn_fid,
        shap_sparsity=shap_sparsity,
        edge_sparsity=edge_sparsity,
        shap_stability=shap_stab,
        gnn_stability=gnn_stab,
        # agreement=agreement,
        explanation=gnn_output["explanation"]
    )

    print("✅ Scorecard computed:")
    print("   metadata: ", scorecard["metadata"])
    print("   tabular metrics: ", scorecard["tabular_metrics"])
    print("   structural metrics: ", scorecard["structural_metrics"])
    # print("   consensus: ", scorecard["consensus"])

    # ==================================================
    # 7. BAC SCORE 
    # ==================================================
    bac_result = compute_enhanced_bac_score(
        tabular_data=scorecard["tabular_metrics"],
        structural_data=scorecard["structural_metrics"],
        # consensus=scorecard["consensus"]
    )

    scorecard["bac_score"] = bac_result["bac_score"]
    scorecard["trust"] = format_bac_for_ui(bac_result)

    print("✅ BAC score computed:")
    print("   bac score: ", scorecard["bac_score"])
    print("   trust: ", scorecard["trust"])

    # ==================================================
    # 8A. MATERIALIZE GRAPH FOR UI (Cytoscape)
    # ==================================================
    graph_data = materialize_graph_for_ui(
        subgraph=gnn_output["subgraph"],
        explanation=gnn_output["explanation"],
        tx_subset=gnn_output["tx_subset"],
        acc_subset=gnn_output["acc_subset"],
        anchor_local_idx=gnn_output["anchor_local_idx"],
        account_encoder=account_encoder,
    )

    print("✅ UI Graph data loaded")

    # ==================================================
    # 8B. DETECT FRAUD PATTERN
    # ==================================================
    
    pattern_info = detect_structural_pattern(
        explanation=gnn_output["explanation"],
        subgraph=gnn_output["subgraph"],
        tx_subset=gnn_output["tx_subset"],
        acc_subset=gnn_output["acc_subset"],
        account_encoder=account_encoder,
        anchor_local_idx=gnn_output["anchor_local_idx"],
    )

    print("✅ Fraud pattern detected")

    # ==================================================
    # 8C. GENERATE ENHANCED SYSTEM EXPLANATION
    # ==================================================

    system_explanation = generate_investigator_narrative(
        anchor_tx_idx=anchor_tx_idx,
        fraud_prob=anchor_prob,
        tx_amount=tx_amount,
        tx_hour=tx_hour,
        features_df=feat_df,
        pattern_info=pattern_info,
        scorecard=scorecard
    )

    print("✅ System explanation generated")

    # ==================================================
    # 9. ASSEMBLE FINAL SCORECARD
    # ==================================================
    scorecard["explanations"] = {
        "shap_features": shap_feat_df.to_dict("records"),
        "gnn_features": gnn_feat_df.to_dict("records")
    }
    scorecard["features"] = feat_df.to_dict("records")

    scorecard["graph"] = graph_data
    scorecard["pattern"] = pattern_info
    scorecard["system_explanation"] = system_explanation

    print("✅ Final scorecard assembled")

    return scorecard
