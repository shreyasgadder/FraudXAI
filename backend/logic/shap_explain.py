import shap
import pandas as pd
import numpy as np

def run_shap(
    shap_explainer,
    tx_xgb_df,
    anchor_tx_idx
):
    X_anchor = tx_xgb_df.iloc[[anchor_tx_idx]]

    shap_values = shap_explainer(X_anchor)
    shap_df = pd.DataFrame({
        "Feature": tx_xgb_df.columns,
        "SHAP_Value": shap_values.values[0, :, 1]
    }).sort_values("SHAP_Value", ascending=False)

    return {
        "shap_values": shap_values,
        "X_anchor": X_anchor,
        "shap_df": shap_df
    }
