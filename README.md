<div align="center">

# 🔧 SBD Demand Forecasting & AI Supply-Chain Agent

**Predict weekly demand, flag stockout risks before they happen, and let an autonomous AI agent write the weekly reorder report — for Stanley Black & Decker's DEWALT and Stanley tool brands.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EB5E28?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-2.1-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.12-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-D97757?style=for-the-badge)](https://www.anthropic.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-FFCB2E?style=for-the-badge)](https://www.trychroma.com/)
[![MCP](https://img.shields.io/badge/MCP-Tool_Server-5A45FF?style=for-the-badge)](https://modelcontextprotocol.io/)

</div>

---

## 📖 Overview

Stanley Black & Decker ships power tools from four warehouses (**Atlanta, Chicago, Dallas, Newark**). Forecasting demand too low means stockouts and lost sales; too high means cash tied up in inventory. This project solves that end-to-end: it explores three years of weekly sales data, trains an **XGBoost** model that beats the legacy Excel baseline, and then exposes those predictions through an **AI layer** — a RAG pipeline and an **MCP tool server** — so planners can ask questions in plain English and an **autonomous Claude agent** can check every warehouse and produce a complete weekly reorder report on its own. The result is a working bridge from raw data → machine-learning model → AI agent.

> 📌 **Result:** XGBoost cut forecast error from a **21.1%** baseline MAPE to **12.8%** — an 8.3-point improvement, comfortably past the 15% deployment target.

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| **Data & Analysis** | pandas · NumPy · Matplotlib · seaborn |
| **Machine Learning** | XGBoost · scikit-learn (MAE / MAPE) · MLflow · pickle |
| **AI / LLM** | Claude (Anthropic SDK) · ChromaDB (RAG) |
| **Serving & API** | FastAPI · Uvicorn · requests |
| **Config** | python-dotenv |

---

## 🏗️ Architecture

```
   data/sbd_sales_data.csv  (raw weekly sales — single source of truth)
            │
            ▼
   ┌─────────────────┐   trains    ┌────────────────────────┐
   │  SPRINT 2       │ ──────────▶ │  XGBoost model (.pkl)   │
   │  Train XGBoost  │             │  registered in MLflow   │
   └─────────────────┘             └────────────┬───────────┘
                                                │
                                                ▼
                          ┌──────────────────────────────────────┐
                          │  generate_forecasts.py  (THE BRIDGE)  │
                          │  load model → 4-week recursive forecast│
                          │  → weeks_of_supply + risk per SKU      │
                          └───────────────────┬───────────────────┘
                                              │ writes
                                              ▼
                          ┌──────────────────────────────────────┐
                          │  data/xgboost_forecasts.csv            │
                          │  (real predictions, 80 SKU/warehouse) │
                          └───────────────────┬───────────────────┘
                                              │ read on startup
                                              ▼
                          ┌──────────────────────────────────────┐
                          │  sprint5_mcp_server.py  (FastAPI)      │
                          │  4 tools served at :8000/tools/...     │
                          └───────────────────┬───────────────────┘
                                              │ HTTP
                              ┌───────────────┴────────────────┐
                              ▼                                ▼
                  ┌──────────────────────┐        ┌───────────────────────┐
                  │ sprint5_claude_mcp.py│        │  sprint6_agent.py      │
                  │ developer-orchestrated│        │  Claude picks tools    │
                  │ Q&A with Claude       │        │  in a loop → REPORT    │
                  └──────────────────────┘        └────────────────────────┘

   ── Parallel path ──
   data/sbd_sales_data.csv ──▶ sprint4_rag.py ──▶ ChromaDB (./vector_db)
                                question ──▶ semantic search ──▶ Claude ──▶ answer
```

**Data flow:** raw sales → trained XGBoost model → MLflow registry → `generate_forecasts.py` → `xgboost_forecasts.csv` → MCP tool server → Claude / autonomous agent.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- An Anthropic API key

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Claude API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### Run the pipeline

```bash
# 1. Explore data & train the model  (run the notebooks top-to-bottom)
#    notebooks/sprint1_eda.ipynb   →  charts in outputs/
#    notebooks/sprint2_model.ipynb →  model in models/

# 2. Generate real forecasts from the trained model
python3 generate_forecasts.py          # → data/xgboost_forecasts.csv

# 3. (Optional) Try the RAG pipeline
python3 sprint4_rag.py

# 4. Start the MCP tool server  (Terminal A)
uvicorn sprint5_mcp_server:app --reload --port 8000
#    Interactive docs: http://localhost:8000/docs

# 5. Ask Claude with live MCP data  (Terminal B)
python3 sprint5_claude_mcp.py

# 6. Run the autonomous agent  (server must be running)
python3 sprint6_agent.py
```

> 💡 Steps 5 & 6 require the MCP server (step 4) to be running on port 8000. Re-run step 2 whenever the model or data changes, then restart the server.

---

## 📦 What Each Sprint Built

| Sprint | File | What it delivers |
|:------:|------|------------------|
| **1 · EDA** | `notebooks/sprint1_eda.ipynb` | Explores 3 years of sales data and surfaces the key drivers — **seasonality** (spring & holiday peaks), a **~47.5% promo lift**, Chicago as the top warehouse, and the highest-volume DEWALT SKUs. Saves 5 charts to `outputs/`. |
| **2 · Model** | `notebooks/sprint2_model.ipynb` | Engineers lag/rolling features, splits time-series **by date**, and trains an **XGBoost** forecaster that beats the rolling-average baseline (**21.1% → 12.8% MAPE**). Exports the model + feature list + metrics. |
| **3 · MLflow** | `notebooks/sprint3_mlflow.ipynb` | Experiment tracking & model registry scaffolding (`mlflow.db`, `mlruns/`). *Placeholder notebook — registry name is consumed by the bridge below.* |
| **🔗 Bridge** | `generate_forecasts.py` | Loads the XGBoost model (MLflow registry, with **pickle fallback**) and produces a **4-week recursive forecast** for all 80 SKU/warehouse combos → `data/xgboost_forecasts.csv`. |
| **4 · RAG** | `sprint4_rag.py` | Builds a **ChromaDB** vector store of every SKU and runs a full **Retrieval-Augmented Generation** pipeline — ask any question in plain English, get a grounded Claude answer. |
| **5 · MCP** | `sprint5_mcp_server.py`<br>`sprint5_claude_mcp.py` | A **FastAPI MCP server** exposing 4 safe tools (`get_stockout_risk`, `get_sku_forecast`, `get_sales_history`, `get_warehouse_summary`) backed by the XGBoost forecasts, plus a client that has Claude answer questions from live tool data. |
| **6 · Agent** | `sprint6_agent.py` | An **autonomous agent** — given a goal, Claude decides which tools to call, loops until it has enough information, and writes a complete **Weekly Reorder Report** (executive summary, reorder list, per-warehouse actions, priorities). |

---

## 📁 Project Structure

```
sbd_project/
├── data/
│   ├── sbd_sales_data.csv          # raw weekly sales (source of truth)
│   └── xgboost_forecasts.csv       # model predictions (from the bridge)
├── notebooks/
│   ├── sprint1_eda.ipynb           # exploratory data analysis
│   ├── sprint2_model.ipynb         # XGBoost training
│   └── sprint3_mlflow.ipynb        # MLflow tracking (placeholder)
├── models/                         # trained model + feature list + metrics
├── outputs/                        # EDA & feature-importance charts
├── vector_db/                      # persisted ChromaDB store
├── generate_forecasts.py           # 🔗 bridge: model → forecasts
├── sprint4_rag.py                  # RAG pipeline
├── sprint5_mcp_server.py           # FastAPI MCP tool server
├── sprint5_claude_mcp.py           # Claude + MCP client
├── sprint6_agent.py                # autonomous supply-chain agent
├── requirements.txt
└── PROJECT_DOCUMENTATION.md         # full technical deep-dive
```

---

## 📚 Documentation

For a complete technical walkthrough — every file, function, and design decision explained in plain English — see **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**.

---

<div align="center">

*Built as a 6-sprint journey from raw data to an autonomous AI agent.* 🚀

</div>
