import copy
import torch
import pandas as pd

def tabular_fidelity(
    model_predict_fn,
    X_anchor,
    shap_df,
    train_df,
    k_list
):
    base_prob = float(model_predict_fn(X_anchor)[0])
    medians = train_df.median()
    rows = []

    for k in k_list:
        x_ablate = X_anchor.copy()
        for f in shap_df.head(k)["Feature"]:
            x_ablate[f] = medians[f]

        ablated_prob = float(model_predict_fn(x_ablate)[0])
        drop = base_prob - ablated_prob

        rows.append({
            "K": k,
            "Base_Prob": base_prob,
            "Ablated_Prob": ablated_prob,
            "Drop": drop,
            "Fidelity_Score": drop / base_prob if base_prob > 0 else 0
        })

    return pd.DataFrame(rows)


def gnn_fidelity(wrapper, subgraph, explanation, anchor_local_idx, top_k):
    with torch.no_grad():
        base_prob = torch.sigmoid(
            wrapper(subgraph.x_dict, subgraph.edge_index_dict)
        )[anchor_local_idx].item()

    ablated_edges = copy.deepcopy(subgraph.edge_index_dict)

    for etype, mask in explanation.edge_mask_dict.items():
        if mask is None or mask.numel() == 0:
            continue

        if mask.numel() <= top_k:
            ablated_edges[etype] = torch.empty((2, 0), dtype=torch.long)
        else:
            _, top_idx = torch.topk(mask, top_k)
            keep = ~torch.isin(torch.arange(mask.numel()), top_idx)
            ablated_edges[etype] = ablated_edges[etype][:, keep]

    with torch.no_grad():
        ablated_prob = torch.sigmoid(
            wrapper(subgraph.x_dict, ablated_edges)
        )[anchor_local_idx].item()

    drop = base_prob - ablated_prob

    return {
        "Base_Prob": base_prob,
        "Ablated_Prob": ablated_prob,
        "Drop": drop,
        "Fidelity_Score": drop / base_prob if base_prob > 0 else 0
    }
