"""
FraudXAI Phase-2B Backend Server (AUTHORITATIVE)

This is the main FastAPI application that serves the fraud investigation dashboard.

CRITICAL RULES (DO NOT VIOLATE):
- Load artifacts ONCE at startup
- Never precompute explanations
- Support partial failure without crashing
- Use Pydantic v2 STRICT schemas
- Gemini 2.5 Flash backend-only
"""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shap
from google import genai
from google.genai import types

from backend.logic.artifact_loader import load_artifacts
from backend.logic.gnn_explain import run_gnn_explainer
from backend.logic.shap_explain import run_shap
from backend.logic.explain_orchestrator import run_full_explanation_pipeline
from backend.logic.gnn_model import EdgeClassifier


# ============================================================
# PYDANTIC V2 SCHEMAS (STRICT)
# ============================================================

class ExplainRequest(BaseModel):
    """Request schema for explanation endpoint"""
    transaction_id: int = Field(..., description="Transaction ID to explain")


class MetadataResponse(BaseModel):
    """Transaction metadata and risk assessment"""
    transaction_id: int
    fraud_probability: float
    risk_level: str
    decision_delta: float


class ConsensusResponse(BaseModel):
    """Agreement between SHAP and GNN explanations"""
    agreement_strength: str
    agreement_ratio: float
    agreement_count: int
    features: List[str]


class ExplainResponse(BaseModel):
    """Complete explanation response (EXACT SCHEMA)"""
    metadata: MetadataResponse
    bac_score: float
    tabular_metrics: dict
    structural_metrics: dict
    explanations: dict
    warnings: List[str] = Field(default_factory=list)
    
    # NEW: Enhanced GNN explainer fields
    graph: Optional[dict] = None  # Cytoscape elements
    pattern: Optional[dict] = None  # Fraud pattern detection
    system_explanation: Optional[str] = None  # Enhanced narrative
    trust: Optional[dict] = None  # Formatted BAC with subgraph quality
    features: Optional[List[dict]] = None  # Unified feature importance (SHAP + GNN)


# ============================================================
# APPLICATION STATE
# ============================================================

class AppState:
    """Global application state (loaded once at startup)"""
    def __init__(self):
        self.artifacts = None
        self.model = None
        self.device = None
        self.shap_explainer = None  
        self.test_data = None


app_state = AppState()


# ============================================================
# STARTUP/SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts and model at startup"""
    print("\n" + "="*60)
    print("🚀 FraudXAI Phase-2B Backend Starting...")
    print("="*60 + "\n")
    
    # Set device
    app_state.device = torch.device("cpu")
    print(f"📍 Device: {app_state.device}\n")
    
    # Load artifacts using artifact_loader (handles download if needed)
    app_state.artifacts = load_artifacts(device="cpu")
    
    # Instantiate GNN model using EdgeClassifier from gnn_model.py
    model_assets = app_state.artifacts["model_assets"]
    hyperparams = model_assets["hyperparams"]
    graph_context = app_state.artifacts["graph_context"]
    
    app_state.model = EdgeClassifier(
        tx_in=hyperparams["in_channels_tx"],
        acc_in=hyperparams["in_channels_acc"],
        hidden_dim=hyperparams["hidden_channels"],
        conv_layers=hyperparams["conv_layers"],
        dropout=hyperparams["dropout"],
        metadata=graph_context["hetero_data"].metadata()
    )
    
    # Load state_dict into model with strict=True (MANDATORY)
    app_state.model.load_state_dict(model_assets["state_dict"], strict=True)
    app_state.model.to(app_state.device)
    app_state.model.eval()
    
    print("\n✅ GNN model loaded and ready")
    print(f"   - Account features: {hyperparams['in_channels_acc']}")
    print(f"   - Transaction features: {hyperparams['in_channels_tx']}")
    print(f"   - Hidden dimensions: {hyperparams['hidden_channels']}")

    tx_xgb_df = graph_context["tx_xgb_df"]
    
    # SHAP EXPLAINER     
    app_state.shap_explainer = shap.Explainer(
        app_state.artifacts["xgb_model"].predict_proba,
        tx_xgb_df.sample(1000, random_state=42),
        feature_names=tx_xgb_df.columns
    )

    print("\n✅ SHAP explainer loaded and ready")

    #  Load test data
    app_state.test_data = pd.read_parquet(app_state.artifacts["ui_test_data_path"])
    print("\n✅ Test data loaded and ready")

    # Configure Gemini (if API key available)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        # genai.configure(api_key=gemini_key)
        print("\n✅ GEMINI_API_KEY found!!")
    else:
        print("\n⚠️  GEMINI_API_KEY not set - narrative generation will be unavailable")
    
    print("\n" + "="*60)
    print("✅ Backend Ready - Listening for requests")
    print("="*60 + "\n")
    
    yield
    
    print("\n🛑 Shutting down FraudXAI backend...")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FraudXAI Phase-2B API",
    description="Production-grade fraud investigation dashboard backend",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "FraudXAI Phase-2B",
        "version": "2.0.0"
    }


@app.get("/transactions")
async def get_transactions(
    limit: int = 20,
    offset: int = 0,
    transaction_id: Optional[int] = None
):
    """
    Get paginated transaction data from UI test dataset.
    Filters to show only medium+ risk transactions for analyst focus.
    
    Args:
        limit: Number of rows to return (default 20)
        offset: Starting row offset (default 0)
        transaction_id: Optional filter by transaction ID
    """
    try:
        df = app_state.test_data
        
        # Rename columns to match frontend expectations
        df = df.rename(columns={
            "global_tx_idx": "transaction_id",
            "nameOrig": "sender_id",
            "nameDest": "receiver_id",
            "gnn_prob": "fraud_probability"
        })
        
        # Filter UI dataset using threshold.medium (REQUIRED)
        model_assets = app_state.artifacts["model_assets"]
        medium_threshold = model_assets["thresholds"]["medium"]
        df = df[df["fraud_probability"] >= medium_threshold]
        
        # Filter by transaction_id if provided
        if transaction_id is not None:
            df = df[df["transaction_id"] == transaction_id]
        
        # Pagination
        total = len(df)
        df_page = df.iloc[offset:offset + limit]
        
        # Select only columns needed for UI (exclude isFraud)
        ui_columns = ["transaction_id", "hour_of_day", "type", "amount", 
                      "sender_id", "receiver_id", "fraud_probability"]
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "transactions": df_page[ui_columns].to_dict(orient="records")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load transactions: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain_transaction(request: ExplainRequest):
    """
    Generate comprehensive explanation for a transaction.
    
    This endpoint orchestrates the full explanation pipeline:
    1. GNNExplainer
    2. SHAP
    3. Materialization
    4. Faithfulness, Sparsity, Stability, Agreement
    5. Scorecard + BAC Score
    
    Supports partial failure - if SHAP fails, continues with GNN only.
    """
    warnings = []
    
    try:
        # Extract artifacts
        graph_context = app_state.artifacts["graph_context"]
        hetero_data = graph_context["hetero_data"]
        account_encoder = graph_context["account_encoder"]
        tx_xgb_df = graph_context["tx_xgb_df"]
        test_idx_map = graph_context["test_idx_map"]
        
        xgb_model = app_state.artifacts["xgb_model"]
        model_assets = app_state.artifacts["model_assets"]
        
        tx_id = request.transaction_id
        
        # CRITICAL: tx_id IS the global_tx_idx (as shown in UI from ui_test_data.parquet)
        # The test_idx_map maps global_idx -> local_test_position, but GNN needs global indices
        # to access data['transaction'].sender[global_idx] correctly
        if tx_id not in test_idx_map:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {tx_id} not found in test dataset"
            )
        
        # Use tx_id directly as anchor - it IS the global tensor index
        anchor_tx_idx = tx_id
        
        # Get anchor_prob from ui_test_data.parquet
        ui_df = app_state.test_data
        ui_df = ui_df.rename(columns={"global_tx_idx": "transaction_id", "gnn_prob": "fraud_probability"})
        anchor_row = ui_df[ui_df["transaction_id"] == tx_id]

        if anchor_row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {tx_id} not found in UI dataset"
            )
        
        anchor_prob = float(anchor_row.iloc[0]["fraud_probability"])
        medium_threshold = model_assets["thresholds"]["medium"]
        
        # ===== 1. GNN EXPLAINER =====
        print(f"\n🔍 Explaining transaction {tx_id} (global_idx={anchor_tx_idx})")

        gnn_output = run_gnn_explainer(
            model=app_state.model,
            data=hetero_data,
            anchor_tx_idx=anchor_tx_idx,
            anchor_pred=anchor_prob,
            tx_feature_names=model_assets["feature_names"],
            account_encoder=account_encoder,
            medium_threshold=medium_threshold,
            device=app_state.device
        )
        
        print("✅ GNN explanation complete")
        
        # ===== 2. SHAP EXPLAINER =====

        shap_output = run_shap(
            shap_explainer=app_state.shap_explainer,
            tx_xgb_df=tx_xgb_df,
            anchor_tx_idx=anchor_tx_idx
        )
        print("✅ SHAP explanation complete")
        
        # ===== 3. FULL PIPELINE =====
        # Get transaction metadata for system explanation
        tx_amount = float(anchor_row.iloc[0].get("amount", 0))
        tx_hour = int(anchor_row.iloc[0].get("hour_of_day", 0))
        
        scorecard = run_full_explanation_pipeline(
            anchor_tx_idx=tx_id,
            shap_output=shap_output,
            gnn_output=gnn_output,
            xgb_model=xgb_model,
            shap_explainer=app_state.shap_explainer,
            tx_xgb_df=tx_xgb_df,
            train_df=tx_xgb_df,
            tx_feature_names=model_assets["feature_names"],
            account_encoder=account_encoder,
            threshold_dict=model_assets["thresholds"],
            anchor_prob=anchor_prob,
            graph_context=graph_context,
            tx_amount=tx_amount,
            tx_hour=tx_hour
        )
        
        # Add warnings
        scorecard["warnings"] = warnings
        
        print(f"✅ Explanation complete - BAC Score: {scorecard['bac_score']}")
        
        return scorecard
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


class NarrativeRequest(BaseModel):
    """Request schema for AI narrative generation with full scorecard"""
    transaction_id: int
    metadata: dict
    explanations: dict
    # NEW: Enhanced context fields
    pattern: Optional[dict] = None
    trust: Optional[dict] = None
    system_explanation: Optional[str] = None


@app.post("/generate-narrative")
async def generate_narrative(request: NarrativeRequest):
    """
    Generate AI narrative using Gemini 2.5 Flash (BACKEND ONLY).
    
    This endpoint must be called from the frontend - no direct Gemini access allowed.
    """
    try:
        # Check if Gemini is configured
        if not os.getenv("GEMINI_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="Gemini API not configured - set GEMINI_API_KEY environment variable"
            )
        
        # Build context for Gemini with FULL scorecard
        # Format unified features (from feat_df)
        features_str = "Not available"
        if request.explanations.get('features'):
            features_str = chr(10).join(
                f"- {f['Feature']}: {f['Importance']:.3f}" 
                for f in request.explanations.get('features', [])[:8]  # Top 8
            )
        
        # Format network context from full graph structure
        graph_context_str = "Not available"
        if request.metadata.get('graph'):
            graph = request.metadata['graph']
            stats = graph.get('stats', {})
            
            # Build graph structure summary
            graph_lines = [
                f"Graph Structure:",
                f"- {stats.get('num_accounts', 0)} accounts",
                f"- {stats.get('num_transactions', 0)} transactions",
                f"- {stats.get('num_edges', 0)} edges"
            ]
            
            # Add key network elements if available
            elements = graph.get('elements', {})
            nodes = elements.get('nodes', [])
            edges = elements.get('edges', [])
            
            if nodes:
                # Find anchor transaction and connected accounts
                anchor_nodes = [n for n in nodes if n.get('data', {}).get('is_anchor')]
                account_nodes = [n for n in nodes if n.get('data', {}).get('node_type') == 'account']
                
                if anchor_nodes:
                    anchor_id = anchor_nodes[0].get('data', {}).get('id', 'Unknown')
                    graph_lines.append(f"\nKey Network Elements:")
                    graph_lines.append(f"- Anchor Transaction: {anchor_id}")
                
                if account_nodes:
                    account_ids = [n.get('data', {}).get('id') for n in account_nodes[:3]]
                    graph_lines.append(f"- Connected Accounts: {', '.join(account_ids)}")
                
                if edges:
                    graph_lines.append(f"- Transaction Relationships: {len(edges)} connections")
            
            graph_context_str = chr(10).join(graph_lines)
        
        # Format pattern info if available
        pattern_str = "No specific pattern detected"
        if request.pattern:
            pattern_str = f"""Pattern Type: {request.pattern.get('pattern_type', 'None')}
Confidence: {request.pattern.get('confidence', 0):.0%}
Evidence: {', '.join(request.pattern.get('evidence', [])[:3])}
High-risk accounts: {', '.join(request.pattern.get('high_risk_accounts', [])[:3]) or 'None'}"""
        
        # Format trust info if available
        trust_str = "Not available"
        if request.trust:
            trust_str = f"Trust Level: {request.trust.get('label', 'Unknown')} ({request.trust.get('bac_score', 0):.1f}%)"
        
        print("Generative AI Narrative parameters:")
        print("   Transaction ID:", request.transaction_id)
        print("   Trust Level:", trust_str)
        print("   Pattern:", pattern_str)
        print("   Features:", features_str)
        print("   Graph Context:", graph_context_str)
        # Build comprehensive prompt
        prompt = f"""
You are a Senior Financial Crime Investigator looking at a flagged transaction. 
Your goal is to provide a concise, high-impact assessment for a Stake Holders.
DO NOT invent facts. ONLY use the information provided.

**CONTEXT:**
Transaction ID: {request.transaction_id}
Fraud Probability: {request.metadata.get('fraud_probability', 0):.1%}
Risk Level: {request.metadata.get('risk_level', 'Unknown')}
{trust_str}

**EVIDENCE:**
[Detected Pattern]
{pattern_str}

[Key Feature Drivers]
{features_str}

[Network Graph]
{graph_context_str}

**INSTRUCTIONS:**
1. **Do not just repeat the data.** Synthesize the features and patterns to explain the *behavior*.
2. Connect the dots: specific accounts + specific amounts + patterns = what story?
3. Use a direct, professional, investigative tone.
4. Output Markdown with the following structure:
   - **🚨 Risk Assessment**: One short sentence defining the core threat.
   - **🔍 Analysis**: 2 sentences explaining the "How" and "Why" by correlating features with the pattern.
   - **🛡️ Action**: A direct command on what to investigate next.
"""
        
        # Call Gemini 2.5 Flash with standard google-generativeai API
        client = genai.Client()
        model_name = "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert fraud investigator. You prefer bullet points and concise, actionable insights over long paragraphs.",
                temperature=0.2
            ),
            contents=prompt
        )
        
        return {
            "narrative": response.text,
            "model": model_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Narrative generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
