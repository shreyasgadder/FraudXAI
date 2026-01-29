# Explainable Fraud Detection in Financial Transactions Using GNN

**M. Tech Data Science & Engineering Dissertation Project | BITS Pilani (WILP)**

## 📌 Project Overview

**FraudXAI** is an advanced Explainable AI (XAI) Dashboard designed to detect and explain fraudulent financial transactions. Unlike traditional "black-box" models that only flag transactions, FraudXAI provides a comprehensive, multi-layer explanation of *why* a transaction is suspicious.

It combines:

1. **Heterogeneous Graph Neural Networks (HeteroGNN):** Captures complex, multi-hop money flow patterns (Structural Analysis) using PyTorch Geometric by modeling accounts and transactions as distinct node types.
2. **Robustness Engine (Focal Loss & TBCV):** Implements Binary Focal Loss to handle the extreme class imbalance (Fraud $\approx$ 0.17%) and uses Time-Based Cross Validation to ensure the model remains stable against concept drift.
3. **Tabular Analysis (SHAP):** Identifies behavioral anomalies in transaction features via a stacked XGBoost prior, providing a "tabular baseline" for every prediction.
4. **GNNExplainer (Structural Inference):** Acts as the "structural lens" by identifying the specific k-hop subgraph edges and nodes that maximize the mutual information for a fraud prediction.
5. **BAC (Behavior-Attribution-Consistency) Score:** A mathematical framework developed to quantify the trustworthiness of the AI's explanation by measuring its faithfulness, sparsity, and stability.
6. **Deterministic Narrative Engine:** A rule-based algorithm that automatically categorizes structural anomalies into known fraud typologies (e.g., "Receiver Aggregation" or "Sender Fan-Out") for audit-ready reporting.
7. **Generative AI (Google Gemini):** Translates complex technical metrics (SHAP values, Jaccard similarities, and GNN weights) into polished, natural language summaries for non-technical stakeholders.
---

## 🏗️ System Architecture

The project follows a modular pipeline designed for production-grade transparency:

* **Backend:** Python **FastAPI** serving REST endpoints and managing the XAI orchestrator.
* **Frontend:** **React.js** dashboard featuring interactive graph visualizations (Cytoscape.js).
* **ML Core:**
* **Graph Model:** `SAGEConv` (GraphSAGE) architecture optimized with **Focal Loss** to handle extreme class imbalance.
* **Tabular Model:** XGBoost for feature stacking and initial probability priors.
* **Explainability:** Post-hoc fusion of **GNNExplainer** and **KernelSHAP**.
* **Narrative Engine:** Hybrid system using deterministic rule-based logic and **Gemini 2.5 Flash**.

---

## 🚀 Getting Started

### Prerequisites

* **Python:** 3.9+ (3.12 recommended)
* **Node.js:** 18+ & npm
* **GPU:** NVIDIA T4 or better (recommended for model training)
* **API Key:** Google Gemini API Key

### 1. Artifact Generation (The ML Pipeline)

Before running the dashboard, you must generate the model artifacts using the provided notebook:

1. Open `notebook/FraudXAI-MLOps-Pipeline.ipynb` in Kaggle.
2. Ensure the environment has a **GPU T4** enabled.
3. Run all cells to:
* Preprocess the **PaySim** dataset.
* Train the XGBoost and HeteroGNN models.
* Export artifacts (`model_assets.pt`, `graph_context.pkl`, `xgb_model.pkl`, `ui_test_data.parquet`) to your Kaggle Model registry via `kagglehub`.

### 2. Backend Configuration & Setup

**Important:** You must point the backend to your specific artifact version.

1. Open `backend/logic/artifact_loader.py`.
2. Update the `KAGGLE_MODEL_HANDLE` to match your exported Kaggle model:
```python
# Example slug from fraudxai-train.py
KAGGLE_MODEL_HANDLE = "your_username/fraud-gnn-xai/pyTorch/v5" 

```

3. Install dependencies:
```bash
cd backend
pip install -r requirements.txt

```

4. Set your Gemini API key:
* **Windows:** `$env:GEMINI_API_KEY="your_api_key"`
* **Linux/Mac:** `export GEMINI_API_KEY="your_api_key"`

5. Start the server:
```bash
python -m uvicorn main:app --reload --port 8000

```

### 3. Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
npm install

```


2. Start the development server:
```bash
npm run dev

```


*The dashboard will be available at `http://localhost:5173*`

---

## 📂 Repository Structure

```
📦 FraudXAI
├── 📂 backend                          # FastAPI Application
│   ├── 📂 logic                        # XAI Orchestrator (GNN, SHAP, BAC Score, Narrative)
│   ├── 📂 artifacts                    # Auto-downloaded artifacts
│   └── main.py                         # API Entry Point
├── 📂 frontend                         # React Dashboard
│   ├── 📂 src
│   │   ├── 📂 components               # StructuralLens, BACScoreCard, Narratives
│   │   └── utils                       # Axios API client
├── 📂 notebook                         # MLOps Pipeline
│   └── FraudXAI-MLOps-Pipeline.ipynb   # Model Training & Artifact Export Script
└── README.md                           # Documentation

```

## 🧠 Performance & Results

* **Detection Accuracy:** Achieved an **F1-Score of 0.8272** and **PR-AUC of 0.8710**.
* **Reliability:** **Brier Score of 0.0083**, indicating highly calibrated risk probabilities.
* **Trust Metric (BAC):** Quantifies explanations based on **Faithfulness** (0.5), **Sparsity** (0.25), and **Stability** (0.25).
* **Typology Detection:** Automatically identifies structural patterns like **Receiver Aggregation (Money Mules)**, **Sender Aggregation**.

---

## 🎓 Dissertation Context

This project was developed for the **BITS Pilani WILP M.Tech Data Science Engineering** dissertation. It addresses the "Black Box" dilemma in FinTech by demonstrating how GNNs can be made interpretable for regulatory compliance and audit-ready reporting.

**Author:** Shreyas Gadder
**Institution:** BITS Pilani (WILP)
