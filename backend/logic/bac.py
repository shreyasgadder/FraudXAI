# enhanced_bac.py
"""
Enhanced BAC (Balanced Attribution Confidence) Score
Now includes subgraph quality assessment
"""

import pandas as pd
from typing import Dict


def compute_enhanced_bac_score(
    tabular_data: Dict,
    structural_data: Dict,
    # consensus: Dict,
    weights: Dict = None
) -> Dict:
    """
    Compute BAC score with subgraph quality factor.
    
    Args:
        tabular_data: SHAP metrics (faithfulness, sparsity, stability)
        structural_data: GNN metrics (faithfulness, stability)
        weights: Optional custom weights
    
    Returns:
        {
            'bac_score': float (0-100),
            'components': dict,
            'trust_level': str,
            'warnings': list
        }
    """
    # Default weights (can be overridden)
    if weights is None:
        weights = {
            'faithfulness': 0.50,
            'sparsity': 0.25,
            'stability': 0.25
        }
    
    components = {}
    warnings = []
    
    # ============================================================
    # 1. FAITHFULNESS (Combined SHAP + GNN)
    # ============================================================
    shap_drop = pd.DataFrame(tabular_data["faithfulness"])["Drop"].mean()
    shap_faith = min(shap_drop, 1.0)
    
    gnn_faith = max(structural_data["faithfulness_edges"]["Drop"], 0)
    
    # Weighted combination (SHAP more reliable for faithfulness)
    combined_faith = (0.75 * shap_faith) + (0.25 * gnn_faith)
    components['faithfulness'] = combined_faith
    
    if shap_faith < 0.3:
        warnings.append("Low tabular faithfulness: top features have weak predictive power")
    
    # ============================================================
    # 2. SPARSITY
    # ============================================================
    sparsity = 1.0 - tabular_data["sparsity"]["sparsity_ratio"]
    components['sparsity'] = sparsity
    
    if sparsity < 0.5:
        warnings.append("Low sparsity: explanation relies on many features")
    
    # ============================================================
    # 3. STABILITY (Combined SHAP + GNN)
    # ============================================================
    shap_stab = max(tabular_data["stability"]["rank_stability"]["spearman_corr"], 0)
    gnn_stab = structural_data["gnn_stability"]["jaccard_similarity"]
    
    combined_stability = (0.70 * shap_stab) + (0.30 * gnn_stab)
    components['stability'] = combined_stability
    
    if combined_stability < 0.6:
        warnings.append("Low stability: explanation may vary under perturbation")
    
    # ============================================================
    # 5. COMPUTE WEIGHTED BAC
    # ============================================================
    bac = (
        weights['faithfulness'] * combined_faith +
        weights['sparsity'] * sparsity +
        weights['stability'] * combined_stability
    )
    
    bac_score = round(bac * 100, 2)
    
    # ============================================================
    # 6. TRUST LEVEL MAPPING
    # ============================================================
    if bac_score >= 70:
        trust_level = "HIGH TRUST"
        trust_color = "green"
    elif bac_score >= 50:
        trust_level = "MODERATE TRUST"
        trust_color = "yellow"
    else:
        trust_level = "LOW TRUST"
        trust_color = "red"
    
    return {
        'bac_score': bac_score,
        'trust_level': trust_level,
        'trust_color': trust_color,
        'components': {
            'faithfulness': round(combined_faith, 3),
            'sparsity': round(sparsity, 3),
            'stability': round(combined_stability, 3)
        },
        'warnings': warnings
    }


# ============================================================
# FRONTEND-READY FORMAT
# ============================================================

def format_bac_for_ui(bac_result: Dict) -> Dict:
    """
    Format BAC score for frontend display.
    """
    return {
        'trust_score': bac_result['bac_score'],
        'trust_level': bac_result['trust_level'],
        'trust_color': bac_result['trust_color'],
        'components': bac_result['components'],
        'warnings': bac_result['warnings']
    }