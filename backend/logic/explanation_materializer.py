#######################################################
# Materialize GNN explanations
########################################################

import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np


def materialize_graph_for_ui(
    *,
    subgraph,
    explanation,
    tx_subset,
    acc_subset,
    anchor_local_idx,
    account_encoder,
):
    """
    Converts subgraph + explanation into Cytoscape.js format.

    Guarantees:
    - UI IDs (tx_<id>, C<account>)
    - Explicit semantic roles
    - No pruning / filtering
    - Full transparency of importance values
    """

    elements = {
        "nodes": [],
        "edges": [],
    }

    # --------------------------------------------------
    # 1. NODE IMPORTANCE LOOKUP
    # --------------------------------------------------
    tx_node_mask = explanation.node_mask_dict.get("transaction")
    acc_node_mask = explanation.node_mask_dict.get("account")

    # --------------------------------------------------
    # 2. MATERIALIZE TRANSACTION NODES
    # --------------------------------------------------
    for local_tx_idx, global_tx_idx in enumerate(tx_subset.tolist()):
        is_anchor = local_tx_idx == anchor_local_idx

        importance = (
            float(tx_node_mask[local_tx_idx].sum().item())
            if tx_node_mask is not None
            else 0.0
        )

        elements["nodes"].append({
            "data": {
                "id": f"tx_{global_tx_idx}",
                "label": f"tx_{global_tx_idx}",
                "type": "transaction",
                "transaction_id": int(global_tx_idx),
                "role": "anchor_transaction" if is_anchor else "connected_transaction",
                "importance": importance,
                "is_anchor": is_anchor,
            }
        })

    # --------------------------------------------------
    # 3. MATERIALIZE ACCOUNT NODES
    # --------------------------------------------------
    for local_acc_idx, global_acc_idx in enumerate(acc_subset.tolist()):
        account_id = account_encoder.inverse_transform([global_acc_idx])[0]

        importance = (
            float(acc_node_mask[local_acc_idx].sum().item())
            if acc_node_mask is not None
            else 0.0
        )

        elements["nodes"].append({
            "data": {
                "id": account_id,
                "label": account_id,
                "type": "account",
                "account_id": account_id,
                "role": "account",
                "importance": importance,
                "is_anchor": False,
            }
        })

    # --------------------------------------------------
    # 4. MATERIALIZE EDGES (SENDS / RECEIVES)
    # --------------------------------------------------
    for edge_type, edge_index in subgraph.edge_index_dict.items():
        src_type, relation, dst_type = edge_type
        edge_mask = explanation.edge_mask_dict.get(edge_type)

        for i in range(edge_index.size(1)):
            src_local = edge_index[0, i].item()
            dst_local = edge_index[1, i].item()

            if src_type == "account":
                src_id = account_encoder.inverse_transform(
                    [acc_subset[src_local].item()]
                )[0]
                dst_id = f"tx_{tx_subset[dst_local].item()}"
                role = "sender_edge"

            else:  # transaction -> account
                src_id = f"tx_{tx_subset[src_local].item()}"
                dst_id = account_encoder.inverse_transform(
                    [acc_subset[dst_local].item()]
                )[0]
                role = "receiver_edge"

            weight = (
                float(edge_mask[i].item())
                if edge_mask is not None
                else 0.0
            )

            elements["edges"].append({
                "data": {
                    "id": f"e_{src_id}_{dst_id}_{i}",
                    "source": src_id,
                    "target": dst_id,
                    "relation": relation,
                    "role": role,
                    "weight": weight,
                }
            })

    # --------------------------------------------------
    # 5. STATS (FOR UI / DEBUG)
    # --------------------------------------------------
    stats = {
        "num_nodes": len(elements["nodes"]),
        "num_edges": len(elements["edges"]),
        "num_transactions": subgraph["transaction"].num_nodes,
        "num_accounts": subgraph["account"].num_nodes,
    }

    return {
        "elements": elements,
        "stats": stats,
    }

#######################################################
# Detect structural fraud patterns
#######################################################

def detect_structural_pattern(
    *,
    explanation,
    subgraph,
    tx_subset,
    acc_subset,
    account_encoder,
    anchor_local_idx,
):
    """
    Detects structural fraud patterns from the FULL explained subgraph.
    No edge filtering. Importance is used only for aggregation.
    """

    sender_accounts = []
    receiver_accounts = []
    sender_weights = []
    receiver_weights = []

    # Iterate over all explained edges
    for edge_type, edge_index in subgraph.edge_index_dict.items():
        mask = explanation.edge_mask_dict.get(edge_type)
        if mask is None:
            continue

        src_type, _, dst_type = edge_type

        for i in range(edge_index.size(1)):
            w = float(mask[i].item())

            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()

            if src_type == "account" and dst_type == "transaction":
                acc_id = account_encoder.inverse_transform(
                    [acc_subset[src].item()]
                )[0]
                sender_accounts.append(acc_id)
                sender_weights.append(w)

            elif src_type == "transaction" and dst_type == "account":
                acc_id = account_encoder.inverse_transform(
                    [acc_subset[dst].item()]
                )[0]
                receiver_accounts.append(acc_id)
                receiver_weights.append(w)

    sender_counts = Counter(sender_accounts)
    receiver_counts = Counter(receiver_accounts)

    avg_sender_weight = np.mean(sender_weights) if sender_weights else 0.0
    avg_receiver_weight = np.mean(receiver_weights) if receiver_weights else 0.0

    print("Pattern Detection parameters:")
    print("   sender_counts", sender_counts)
    print("   receiver_counts", receiver_counts)
    print("   avg_sender_weight", avg_sender_weight)
    print("   avg_receiver_weight", avg_receiver_weight)

    top_receiver, rx_count = receiver_counts.most_common(1)[0]
    top_sender, sx_count = sender_counts.most_common(1)[0]

    evidence = []

    # ---- Pattern 1: Receiver aggregation (Money Mule–like)
    if rx_count >= 3 and avg_receiver_weight > avg_sender_weight:
        pattern = "Receiver Aggregation Pattern"
        confidence = min(0.9, 0.6 + rx_count * 0.05)
        evidence.append(
            f"Receiver account {top_receiver} appears repeatedly across connected transactions"
        )
        evidence.append(
            f"Receiver-side structural influence ({avg_receiver_weight:.2f}) exceeds sender-side influence ({avg_sender_weight:.2f})"
        )

    # ---- Pattern 2: Sender burst / fan-out
    elif sx_count >= 3 and avg_sender_weight > avg_receiver_weight:
        pattern = "Sender Fan-Out Pattern"
        confidence = min(0.85, 0.55 + sx_count * 0.05)
        evidence.append(
            f"Sender account {top_sender} initiates multiple structurally influential transactions"
        )

    # ---- Pattern 3: Distributed network
    elif len(sender_counts) > 3 and len(receiver_counts) > 3:
        pattern = "Distributed Network Activity"
        confidence = 0.65
        evidence.append(
            "Structural influence is spread across multiple senders and receivers"
        )

    # ---- Pattern 4: No pattern detected
    else:
        pattern = "Unclear / Isolated Activity"
        confidence = 0.5
        evidence.append(
            "Insufficient structural influence to detect a pattern"
        )

    return {
        "pattern_type": pattern,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "sender_accounts": list(sender_counts.keys()),
        "receiver_accounts": list(receiver_counts.keys()),
    }

#######################################################
# Generate investigator narrative
#######################################################

def generate_investigator_narrative(
    *,
    anchor_tx_idx: int,
    fraud_prob: float,
    tx_amount: float,
    tx_hour: int,
    # shap_features,
    # gnn_features,
    features_df,
    pattern_info,
    scorecard
):
    """
    Investigator-grade unified explanation.
    Explicit, factual, auditable.
    """

    risk_level = scorecard["metadata"]["risk_level"]
    bac_score = scorecard["bac_score"]
    # agreement_info = scorecard["consensus"]

    lines = []

    # --------------------------------------------------
    # 1. Decision Summary
    # --------------------------------------------------
    lines.append(
        f"Transaction {anchor_tx_idx} was classified as {risk_level} "
        f"with {fraud_prob:.2%} fraud probability."
    )

    # --------------------------------------------------
    # 2. Tabular Drivers (SHAP)
    # --------------------------------------------------
    # top_shap = shap_features.iloc[0]
    # lines.append(
    #     f"The primary behavioral driver was '{top_shap['Feature']}', "
    #     f"which showed a strong deviation from normal patterns "
    #     f"(importance score: {top_shap['SHAP_Value']:.2%})."
    # )

    if tx_amount > 100000:
        lines.append(
            f"The transaction amount, ${tx_amount:,.2f}, is unusually large."
        )

    if tx_hour < 6 or tx_hour > 22:
        lines.append(
            f"The transaction occurred at an atypical hour, {tx_hour}:00."
        )

    # --------------------------------------------------
    # 3. Structural Context (GNN)
    # --------------------------------------------------
    # top_gnn = gnn_features.iloc[0]
    # lines.append(
    #     f"From a network perspective, the most influential structural factor "
    #     f"was '{top_gnn['Feature']}' (importance: {top_gnn['GNN_Value']:.2%})."
    # )

    top_features = features_df.iloc[0]
    lines.append(
        f"From a Feature perspective, the most influential factor "
        f"was '{top_features['Feature']}' with {top_features['Importance']:.2%} importance."
    )

    # --------------------------------------------------
    # 4. Pattern Detection
    # --------------------------------------------------
    lines.append(
        f"Structural pattern analysis indicates a "
        f"'{pattern_info['pattern_type']}' "
        f"with {pattern_info['confidence']:.2%} confidence."
    )

    for ev in pattern_info["evidence"]:
        lines.append(f"- {ev}")

    # --------------------------------------------------
    # 5. Trust
    # --------------------------------------------------
    # lines.append(
    #     f"Tabular and network explanations show "
    #     f"{agreement_info['agreement_strength']} agreement "
    #     f"(agreement ratio: {agreement_info['agreement_ratio']:.2%})."
    # )

    lines.append(
        f"The combined explanation trust score (BAC) for this decision is "
        f"{bac_score}%."
    )

    # --------------------------------------------------
    # 6. Recommended Action
    # --------------------------------------------------
    if risk_level == "HIGH RISK":
        lines.append(
            "Recommended action: initiate immediate investigation and "
            "review related accounts and transactions."
        )
    elif risk_level == "MEDIUM RISK":
        lines.append(
            "Recommended action: perform secondary review and monitor "
            "for follow-up activity."
        )
    else:
        lines.append(
            "Recommended action: standard monitoring."
        )

    return "\n".join(lines)
