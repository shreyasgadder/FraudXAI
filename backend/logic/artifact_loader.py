"""
Artifact Loader (Authoritative)

This module is the SINGLE source of truth for loading all model artifacts.
It is intentionally verbose so agents (Antigravity) understand:

- what each artifact is
- why it exists
- how it is used

Rules:
- Download artifacts ONLY if missing
- Load everything ONCE at startup
- Never download per request
"""

import os
import shutil
import json
import torch
import joblib
import xgboost
import kagglehub


# ============================================================
# CONFIGURATION
# ============================================================

KAGGLE_MODEL_HANDLE = "shreyasgadder/fraud-gnn-xai/pyTorch/v5"

ARTIFACT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "artifacts"
)
ARTIFACT_DIR = os.path.abspath(ARTIFACT_DIR)

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# EXPECTED ARTIFACTS (DO NOT RENAME)
# ============================================================

ARTIFACT_FILES = {
    # GNN model weights + architecture metadata
    "model_assets.pt": "GNN weights, architecture hyperparams, risk thresholds",

    # Trained XGBoost model used for SHAP + fidelity
    "xgb_model.pkl": "Tabular fraud model (XGBoost)",

    # Graph + encoders + full tabular data
    "graph_context.pkl": "HeteroData graph, account encoder, full tx_xgb_df",

    # UI search table (TEST split only)
    "ui_test_data.parquet": "Search table for UI (no explanations)",  
}


# ============================================================
# DOWNLOAD IF NEEDED
# ============================================================

def ensure_artifacts_present():
    """
    Downloads artifacts from KaggleHub ONLY if missing locally.
    """
    missing = [
        f for f in ARTIFACT_FILES
        if not os.path.exists(os.path.join(ARTIFACT_DIR, f))
    ]

    if not missing:
        print("✅ All artifacts already present. Skipping download.")
        return

    print("📥 Missing artifacts detected:")
    for f in missing:
        print(f"   - {f}: {ARTIFACT_FILES[f]}")

    print("\n📡 Downloading artifacts from KaggleHub...")
    try:
        # Download to KaggleHub's cache (returns the path)
        downloaded_path = kagglehub.model_download(KAGGLE_MODEL_HANDLE)
        
        print(f"✅ Downloaded to: {downloaded_path}")
        # Copy all files from downloaded path to your artifact dir
        for item in os.listdir(downloaded_path):
            src = os.path.join(downloaded_path, item)
            dst = os.path.join(ARTIFACT_DIR, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"📦 Artifacts copied to: {ARTIFACT_DIR}")
        print(f"📂 Files: {os.listdir(ARTIFACT_DIR)}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")

    print("✅ Artifact download complete.")


# ============================================================
# LOAD ARTIFACTS INTO MEMORY
# ============================================================

def load_artifacts(device: str = "cpu"):
    """
    Loads all artifacts into memory.
    This should be called ONCE at backend startup.
    """

    ensure_artifacts_present()

    print("\n🔧 Loading artifacts into memory...")

    # -----------------------------
    # 1. UI SEARCH DATA
    # -----------------------------
    ui_test_data_path = os.path.join(
        ARTIFACT_DIR,
        "ui_test_data.parquet"
    )
    print("   • ui_test_data.parquet available")

    # -----------------------------
    # 2. GNN MODEL ASSETS
    # -----------------------------
    model_assets = torch.load(
        os.path.join(ARTIFACT_DIR, "model_assets.pt"),
        map_location=device
    )

    # Contents:
    # - state_dict
    # - hyperparams: in_channels_acc, in_channels_tx, hidden_channels
    # - thresholds: risk cutoffs
    # - feature_names: transaction features
    print("   • model_assets.pt loaded")

    # -----------------------------
    # 3. GRAPH CONTEXT
    # -----------------------------
    graph_context = joblib.load(
        os.path.join(ARTIFACT_DIR, "graph_context.pkl")
    )

    # Contents:
    # - hetero_data (CPU)
    # - account_encoder
    # - tx_xgb_df (FULL dataset)
    # - test_idx_map
    print("   • graph_context.pkl loaded")

    # -----------------------------
    # 4. XGBOOST MODEL
    # -----------------------------
    xgb_model = joblib.load(
        os.path.join(ARTIFACT_DIR, "xgb_model.pkl")
    )
    print("   • xgb_model.pkl loaded")

    print("\n✅ All artifacts loaded successfully.")

    return {
        "ui_test_data_path": ui_test_data_path,
        "model_assets": model_assets,
        "graph_context": graph_context,
        "xgb_model": xgb_model
    }
