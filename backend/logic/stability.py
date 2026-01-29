import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from torch_geometric.utils import dropout_edge

# ROBUST SHAP VECTOR EXTRACTION HELPERS

def shap_stability(shap_values, shap_explainer, X_anchor, tx_xgb_df):
    def get_shap_vector(shap_out):
        """
        Returns a 1D numpy array of SHAP attributions.
        Used for initial cosine similarity check.
        """
        if hasattr(shap_out, "values"):
            vals = shap_out.values
        else:
            vals = shap_out
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        return np.asarray(vals).reshape(-1)

    def extract_shap_vector(shap_out, positive_class=1):
        """
        Standardizes SHAP output across different API versions.
        Returns: (values, feature_names)
        """
        if hasattr(shap_out, "values"):  # modern shap.Explanation
            values = shap_out.values
            names = shap_out.feature_names
        elif isinstance(shap_out, list):  # older TreeExplainer API
            values = shap_out[positive_class]
            names = tx_xgb_df.columns
        else:
            values = shap_out
            names = tx_xgb_df.columns
        values = np.array(values)
        # Normalize dimensions to (N,)
        if values.ndim == 3:
            values = values[0, :, positive_class]
        elif values.ndim == 2 and values.shape[1] == 2:
            values = values[:, positive_class]
        elif values.ndim == 2:
            values = values[0]

        return values.astype(float), list(names)
    
    orig = get_shap_vector(shap_values)
    orig_vals, orig_feat_names = extract_shap_vector(shap_values)

    X_noisy = X_anchor.copy()
    X_noisy += np.random.normal(0, 0.01, X_noisy.shape)

    noisy_raw = shap_explainer.shap_values(X_noisy)
    noisy = get_shap_vector(noisy_raw)
    noisy_vals, _ = extract_shap_vector(noisy_raw)

    cos = cosine_similarity(
        orig.reshape(1, -1),
        noisy.reshape(1, -1)
    )[0, 0]

    rank_corr, _ = spearmanr(orig_vals, noisy_vals)

    return {
        "cosine_similarity": float(cos),
        "rank_stability": {
            "spearman_corr": float(rank_corr)
        }
    }


def gnn_stability(explainer, subgraph, anchor_local_idx, target, p=0.05):
    expl_orig = explainer(
        x=subgraph.x_dict,
        edge_index=subgraph.edge_index_dict,
        index=anchor_local_idx,
        target=target
    )

    perturbed = {
        et: dropout_edge(ei, p=p, training=False)[0]
        for et, ei in subgraph.edge_index_dict.items()
    }

    expl_noisy = explainer(
        x=subgraph.x_dict,
        edge_index=perturbed,
        index=anchor_local_idx,
        target=target
    )

    def top_edges(expl, k=5):
        masks = torch.cat([m for m in expl.edge_mask_dict.values() if m is not None])
        return set(torch.topk(masks, min(k, masks.numel())).indices.cpu().numpy())

    A = top_edges(expl_orig)
    B = top_edges(expl_noisy)

    jaccard = len(A & B) / len(A | B)

    return {
        "jaccard_similarity": jaccard,
        "overlap_count": len(A & B)
    }
