# SBD Demand Forecasting Project — Technical Documentation

> A 30-minute onboarding guide for new team members.
> Read top to bottom and you will understand what this project does, how it is built, and how every file connects.

---

## 1. What This Project Does (The Big Picture)

This is a **demand forecasting and inventory-risk system for Stanley Black & Decker (SBD)** — the company behind the **DEWALT** and **Stanley** tool brands.

SBD ships power tools (drills, impact drivers, batteries, etc.) from four warehouses: **Atlanta, Chicago, Dallas, and Newark**. The core business problem is simple to state and expensive to get wrong:

> *"How many of each product will we sell in the coming weeks, and which products are about to run out of stock?"*

If you forecast too low, products sell out and the company loses sales. If you forecast too high, money is tied up in inventory sitting on shelves. The goal is to predict weekly demand accurately and flag **stockout risks** before they happen, so supply-chain planners can reorder in time.

The project was built in **six sprints**, each one a layer on top of the last. It moves from raw data analysis all the way up to an autonomous AI agent that writes a weekly reorder report on its own:

| Sprint | File | What it adds |
|--------|------|--------------|
| 1 | `notebooks/sprint1_eda.ipynb` | Explore the data, find patterns (seasonality, promos, top products) |
| 2 | `notebooks/sprint2_model.ipynb` | Train a machine-learning model that predicts demand |
| 3 | `notebooks/sprint3_mlflow.ipynb` | Track model experiments with MLflow *(placeholder — see notes)* |
| — | `generate_forecasts.py` | **Bridge:** load the XGBoost model and generate real 4-week predictions for every SKU |
| 4 | `sprint4_rag.py` | Let Claude (an AI) answer questions about the data using RAG |
| 5 | `sprint5_mcp_server.py` + `sprint5_claude_mcp.py` | Expose the data as safe "tools" Claude can call |
| 6 | `sprint6_agent.py` | An autonomous agent that uses those tools to write a report |

The first half of the project (Sprints 1–3) is **traditional data science** — analyze data, train a model. The second half (Sprints 4–6) is **AI/LLM engineering** — connect that data to Claude so people can ask questions in plain English and get an AI to act on it. A small bridge script, **`generate_forecasts.py`**, ties the two halves together: it runs the trained model to produce real predictions that the AI layer then serves.

---

## 2. The Data

Everything starts from one file: **`data/sbd_sales_data.csv`** (~12,480 rows of weekly sales).

Each row is one product, in one warehouse, in one week. The columns are:

| Column | Meaning |
|--------|---------|
| `date` | The week (weekly data from 2021 through 2023) |
| `week_of_year` | Week number 1–52 (captures seasonality) |
| `year` | 2021, 2022, or 2023 |
| `warehouse` | Atlanta, Chicago, Dallas, or Newark |
| `brand` | DEWALT or Stanley |
| `sku_id` | The product code, e.g. `DCF899_ImpactDriver` |
| `demand` | Units sold that week (this is what we predict) |
| `inventory` | Units in stock at the end of the week |
| `is_promo` | 1 if there was a promotion that week, 0 if not |

A **SKU** ("Stock Keeping Unit") is just a unique product. The same SKU appears in multiple warehouses, so the natural unit of analysis throughout the whole project is one **SKU + warehouse combination**.

---

## 3. The Tools & Libraries Used

**Data science & modeling**
- **pandas** — load and reshape the CSV data (used everywhere).
- **numpy** — numeric helpers (e.g. clipping negative predictions to zero).
- **matplotlib / seaborn** — charts in the EDA and model notebooks.
- **scikit-learn** — accuracy metrics (MAE, MAPE).
- **XGBoost** — the gradient-boosted-trees model that does the forecasting.
- **MLflow** — experiment tracking (records `mlflow.db` and the `mlruns/` folder).
- **pickle** — saves the trained model to disk as a `.pkl` file.

**AI / LLM layer**
- **anthropic** — the official SDK for calling **Claude** (the LLM that answers questions and acts as the agent).
- **chromadb** — a vector database for RAG (semantic search over the data).
- **FastAPI + uvicorn** — builds and runs the MCP server (a small web API).
- **requests** — the client scripts use this to call the MCP server over HTTP.
- **python-dotenv** — loads the `ANTHROPIC_API_KEY` from the `.env` file.

> Note: `requirements.txt` is a full frozen environment and lists many extra packages (langchain, langgraph, prophet, flask, etc.). Those are installed but **not actually imported** by the seven files documented here — they are leftovers/experiments. The libraries above are the ones the code actually uses.

---

## 4. File-by-File Walkthrough

### 4.1 Sprint 1 — `notebooks/sprint1_eda.ipynb` (Exploratory Data Analysis)

**Purpose:** Understand the data before building anything. EDA = "look at the data, find the patterns, draw the charts." This sprint produces no model — just insights and the PNG charts saved in `outputs/`.

Cell by cell:
1. **Imports** — pandas, numpy, matplotlib, seaborn.
2. **Load & first look** — reads the CSV, prints shape, columns, first rows.
3. **Data quality check** — checks for missing values, prints the date range, lists warehouses/brands, counts SKUs, shows demand statistics. (Always done before any analysis.)
4. **Total weekly demand chart** → `outputs/01_total_demand.png` — total units sold per week across the whole company.
5. **Seasonality chart** → `outputs/02_seasonality.png` — average demand by week-of-year, with highlighted bands for the spring peak (weeks 10–22), holiday peak (weeks 44–52), and January dip (weeks 1–6).
6. **Warehouse demand chart** → `outputs/03_warehouse_demand.png` — average demand per warehouse, ranked.
7. **Promo impact chart** → `outputs/04_promo_impact.png` — compares average demand on promo vs non-promo weeks and computes the **promo lift** (~47.5% more demand during promotions).
8. **Top SKUs chart** → `outputs/05_top_skus.png` — the 10 highest-volume products, colored by brand.
9. **Findings summary** — prints the five key takeaways in plain text.

**The five findings that drive the rest of the project:**
1. **Seasonality** — demand spikes at the same times every year.
2. **November is the biggest spike** (35–50% above average) — must be predicted 4 weeks ahead.
3. **Chicago is the highest-demand warehouse**, Newark the lowest.
4. **Promotions drive ~47.5% more demand** — so `is_promo` must be a model input.
5. **DEWALT Battery, Impact Driver, and Drill Driver** are the top 3 products by volume.

These findings directly decide which features go into the Sprint 2 model.

---

### 4.2 Sprint 2 — `notebooks/sprint2_model.ipynb` (The Forecasting Model)

**Purpose:** Train a machine-learning model that predicts weekly demand, and prove it beats the current method.

Cell by cell:
1. **Imports** — pandas, numpy, matplotlib, sklearn metrics, xgboost.
2. **Load & sort** — reads the CSV and sorts by `sku_id, warehouse, date`. Sorting is *critical* for time-series: if rows are out of order, the lag features (below) will be computed from the wrong weeks.
3. **Feature engineering** — the heart of the notebook. A model can't learn from raw dates, so we turn history into numeric columns, grouped per SKU+warehouse:
   - **Lag features** — `demand_lag1`, `demand_lag2`, `demand_lag4` = demand 1, 2, and 4 weeks ago.
   - **Rolling averages** — `rolling_mean_4`, `rolling_mean_8` = smoothed average demand over the last 4 and 8 weeks.
   - **Rolling std** — `rolling_std_4` = how volatile this product has been recently.
   - The first few weeks of each SKU have no history, so those rows are dropped.
4. **Train/test split** — splits **by date**, never randomly. The last 8 weeks become the test set; everything before is training. (Random splitting would let the model "see the future" and look artificially good.)
5. **Baseline model** — uses the simple rolling-4-week average as predictions. This mimics what the current Excel-based process does. Result: **~21.1% MAPE**. The ML model has to beat this by 15% to be worth deploying.
6. **XGBoost model** — trains `XGBRegressor` (300 trees, learning rate 0.05, max depth 5, 80% subsample, fixed random seed). Predictions are clipped at 0 (demand can't be negative). Result: **~12.8% MAPE**, an 8.3-percentage-point improvement — **target met**.
7. **Feature importance chart** → `outputs/06_feature_importance.png` — shows which inputs the model relied on most.
8. **Save artifacts** — writes three files to `models/`:
   - `xgb_demand_model.pkl` — the trained model (pickled).
   - `feature_list.txt` — the exact 9 features, in order, for whoever deploys the model.
   - `model_results.txt` — the accuracy scores.

**The 9 features the model learns from:** `week_of_year`, `year`, `is_promo`, `demand_lag1`, `demand_lag2`, `demand_lag4`, `rolling_mean_4`, `rolling_mean_8`, `rolling_std_4`.

**Two metrics to know:**
- **MAE** (Mean Absolute Error) — average units off per prediction (~33.8 units).
- **MAPE** (Mean Absolute Percentage Error) — average % off per prediction (lower is better).

---

### 4.3 Sprint 3 — `notebooks/sprint3_mlflow.ipynb` (Experiment Tracking)

**Status: this notebook is currently empty (0 bytes).** It was the planned home for **MLflow** experiment tracking — the practice of logging each model run's parameters and metrics so you can compare experiments over time and register the best model.

The scaffolding for it exists in the repo: `mlflow.db` (the SQLite tracking backend) and the `mlruns/` folder (currently just the empty "Default" experiment, no runs logged). MLflow is also listed in `requirements.txt`.

**For a new team member:** treat this as a known gap. When implemented, this sprint would wrap the Sprint 2 training run in `mlflow.start_run()`, log the XGBoost parameters and the MAPE/MAE metrics, and register `xgb_demand_model.pkl` in the MLflow model registry under the name **`sbd_demand_forecast_xgb`** — which is the name the bridge script below tries to load.

---

### 4.3b — `generate_forecasts.py` (The Bridge: Model → Real Forecasts)

**Purpose:** This is the piece that makes the whole system use *real machine-learning predictions* instead of simple averages. It connects the **Sprint 3 MLflow model** to the **Sprint 5 MCP server**: it loads the trained XGBoost model, generates an actual 4-week-ahead forecast for **every one of the 80 SKU/warehouse combinations**, and saves the results to a CSV the server reads on startup.

It runs in five steps:

**Step 1 — Load the data.** Reads `data/sbd_sales_data.csv` and sorts it by `sku_id, warehouse, date` (same as Sprint 2).

**Step 2 — Rebuild the features.** Recreates the exact same 9 features the model was trained on (lags, rolling means, rolling std). The feature recipe *must* match training or the model would receive inputs it doesn't understand.

**Step 3 — Load the XGBoost model.** Points MLflow at the tracking database (`sqlite:///notebooks/mlflow.db`) and tries to load the registered model `models:/sbd_demand_forecast_xgb/latest` via `mlflow.xgboost.load_model`. **If the registry load fails, it falls back to the pickle file** `models/xgb_demand_model.pkl`. This fallback is why the project still works even though the Sprint 3 notebook is empty.

**Step 4 — Generate predictions.** For each SKU/warehouse, it takes the most recent row of features and rolls the forecast forward **4 weeks**. This is the clever part — an **iterative (recursive) forecast**:
- Predict week 1, clip negatives to 0.
- Feed that prediction back in as next week's `demand_lag1` (shifting the older lags down), and update the rolling mean.
- Repeat for weeks 2, 3, and 4, advancing `week_of_year`/`year` and assuming no promo.

It then sums the four weeks, computes the average weekly forecast, derives **`weeks_of_supply`** (current inventory ÷ avg weekly forecast), and assigns the same **risk label** used everywhere else (HIGH < 16 weeks, MEDIUM < 20, else LOW).

**Step 5 — Save.** Writes everything to **`data/xgboost_forecasts.csv`** — one row per SKU/warehouse with the 4 weekly forecasts, the total, the average, weeks of supply, risk, and a `model` tag (`"XGBoost v1 — MLflow registered"`). It prints a risk breakdown and sample HIGH-risk SKUs.

**How it connects:** run this script **once before starting the MCP server**. The server detects the CSV and serves these real predictions; without it, the server falls back to simple historical averages (see Sprint 5a). Re-run it whenever the model or data changes, then restart the server.

**Run it with:** `python3 generate_forecasts.py`

---

### 4.4 Sprint 4 — `sprint4_rag.py` (RAG: Talk to Your Data with Claude)

**Purpose:** Connect the data to **Claude** so anyone can ask questions in plain English. This script introduces **RAG (Retrieval-Augmented Generation)** — the technique of retrieving relevant data first, then giving it to the LLM so its answers are grounded in real facts instead of guesses.

The script runs in four parts:

**Part 1 — First Claude call.** A "hello world" — sends a simple question to Claude (`claude-sonnet-4-5`) with a system prompt that tells it to act as an SBD supply-chain analyst. Confirms the API key and SDK work.

**Part 2 — Give Claude the real data.** Loads the CSV, builds a `summary` table per SKU+warehouse with `avg_demand`, `last_inventory`, and total promos. It then computes:
- **`weeks_of_supply`** = current inventory ÷ average weekly demand (how many weeks until stockout).
- **`risk`** = `HIGH` if under 16 weeks, `MEDIUM` if under 20, else `LOW`.

The top-5 HIGH-risk SKUs are turned into text and pasted into a prompt asking Claude which products need attention and what to do. (This is "stuffing the context" — manually giving Claude the data.)

**Part 3 — Build the ChromaDB vector database.** This is the RAG infrastructure:
- Creates a persistent ChromaDB client saving to `./vector_db`.
- Deletes any old `sbd_forecasts` collection and recreates it fresh.
- Converts **every** SKU+warehouse row into a rich text "document" (brand, product type, demand, inventory, weeks of supply, risk level, status sentence).
- Stores all documents in ChromaDB. ChromaDB automatically turns each into an **embedding** (a numeric vector capturing its meaning) so it can be searched by similarity later.

**Part 4 — The full RAG pipeline.** The key function:

- **`ask_sbd_data(question)`** — the complete RAG loop:
  1. Search ChromaDB for the 5 documents most relevant to the question (`collection.query`).
  2. Join those documents into one context string.
  3. Build a prompt = retrieved data + the user's question.
  4. Send it to Claude with the analyst system prompt.
  5. Return Claude's answer.

The script then tests it with three questions (e.g. *"Which DEWALT tools are at highest risk?"*).

**Why RAG matters:** instead of giving Claude *all* the data (slow, expensive, and it can't fit), you give it only the few pieces relevant to each question. The answer stays grounded in real numbers.

---

### 4.5 Sprint 5a — `sprint5_mcp_server.py` (The MCP Server / Tool API)

**Purpose:** Expose the forecast data as a set of **safe, structured "tools"** that an AI can call. This is built as an **MCP-style server** using **FastAPI** — essentially a small web API where each endpoint is one tool.

> **What is MCP?** Model Context Protocol — a standard way to give an LLM controlled access to external data and actions through well-defined "tools," instead of letting it touch the raw database. Each tool has clear inputs, clear outputs, and validation.

On startup the server builds its `summary` table — the data behind every tool — using a **two-tier strategy**:
1. **Preferred:** if `data/xgboost_forecasts.csv` exists (produced by `generate_forecasts.py`), the server loads it and uses the **real XGBoost predictions**. It renames `avg_weekly_forecast` to `avg_demand` so the rest of the code is unchanged. You'll see `"Using XGBoost model predictions"` in the logs.
2. **Fallback:** if that file is missing, it falls back to computing simple historical averages and weeks-of-supply directly from `sbd_sales_data.csv` (the original Sprint 4 behavior), logging `"XGBoost forecasts not found — falling back to simple averages"`.

Either way the downstream tools work identically — only the *quality* of the numbers changes. This is what turns the demo into a real ML-backed system.

It exposes **four tools** plus a health check:

- **`get_stockout_risk(warehouse?, weeks_ahead=2, top_n=10)`** — the most-used tool. Returns the top-N SKUs at highest stockout risk, sorted by lowest weeks of supply, optionally filtered to one warehouse. Each item includes risk level, weeks of supply, demand, inventory, and a recommended action. *"What should I reorder this week?"*

- **`get_sku_forecast(sku_id, warehouse)`** — a detailed view of one product in one warehouse: current inventory, weeks of supply, risk, and a simple 4-week-ahead forecast based on its recent average. *"Tell me everything about this one product."*

- **`get_sales_history(sku_id, warehouse, last_n_weeks=12)`** — recent week-by-week history for one SKU, plus summary stats (avg/max/min demand, promo weeks) and a computed **trend** (increasing / decreasing / stable, by comparing the first vs second half of the period). Used to explain *why* a forecast looks the way it does.

- **`get_warehouse_summary(warehouse)`** — overall health of one warehouse: a count of HIGH/MEDIUM/LOW risk SKUs, a **health score** (% of SKUs at LOW risk), the top-3 most urgent SKUs, and an overall status (HEALTHY / MONITOR CLOSELY / ATTENTION NEEDED). *"How is Atlanta doing this week?"*

- **`/` (health check)** — confirms the server is running and lists available tools.

Each tool returns clean JSON and raises a clear **HTTP 404** when a warehouse or SKU isn't found — so the AI gets a helpful error instead of crashing.

**Run it with:** `uvicorn sprint5_mcp_server:app --reload --port 8000` (interactive docs at `http://localhost:8000/docs`).

---

### 4.6 Sprint 5b — `sprint5_claude_mcp.py` (Claude + MCP, Hardcoded)

**Purpose:** Show Claude answering questions using the MCP server's live data — but with the tool calls **wired in by hand** (Claude doesn't choose the tools yet; that's Sprint 6).

Two functions:
- **`call_mcp_tool(tool_name, params)`** — a thin HTTP client that calls a tool on the running MCP server (`http://localhost:8000`) with `requests` and returns the JSON.
- **`ask_claude_with_mcp(question)`** — the demo flow:
  1. Calls `get_warehouse_summary` for all four warehouses.
  2. Calls `get_stockout_risk` for the top-5 urgent SKUs.
  3. Bundles all that into one context string.
  4. Sends context + question to Claude and prints the answer.

It runs two example questions (*"Which warehouse needs the most attention?"* and *"What are the top 3 SKUs to reorder today?"*).

**Key point:** here the **developer** decides which tools to call. The data is live (from the server), but the orchestration is fixed. Sprint 6 hands that decision-making to Claude itself.

**Requires the Sprint 5a server to be running first.**

---

### 4.7 Sprint 6 — `sprint6_agent.py` (The Autonomous Agent)

**Purpose:** The capstone. An **autonomous AI agent** that is given a goal and decides *on its own* which tools to call, in what order, until it has enough information to write a complete **Weekly Reorder Report**.

Built in four steps:

**Step 1 — Tool definitions.** Defines the same four tools (`get_warehouse_summary`, `get_stockout_risk`, `get_sku_forecast`, `get_sales_history`) in **Anthropic's tool-use schema** — each with a name, a description (which tells Claude *when* to use it), and an `input_schema` describing its parameters. Claude reads these descriptions to decide what to call.

**Step 2 — `execute_tool(tool_name, tool_input)`.** When Claude decides to use a tool, this function actually runs it: it calls the MCP server over HTTP (same as Sprint 5b) and returns the result as JSON. It prints each call so you can watch the agent think.

**Step 3 — `run_agent(goal)`.** The **agent loop** — the core idea of the whole sprint:
1. Send the goal + tool definitions to Claude.
2. Claude replies with either a final answer (`stop_reason == "end_turn"`) **or** a request to use one or more tools (`stop_reason == "tool_use"`).
3. If it wants tools: run each one via `execute_tool`, append the results to the conversation, and loop back to step 1.
4. Repeat until Claude produces its final answer — capped at `max_steps = 15` as a safety limit against infinite loops.

The system prompt instructs the agent to always check all four warehouses, start with `get_warehouse_summary`, then drill into risks and forecasts.

**Step 4 — Run it.** Gives the agent a detailed goal (check all warehouses, find HIGH-risk SKUs, get forecasts, and write a report with an executive summary, reorder list, per-warehouse actions, and priority ranking) and prints the final report.

**The difference from Sprint 5:** in Sprint 5 the developer chose the tools; here **Claude chooses them itself**, looping until the job is done. That autonomy is what makes it an "agent."

**Requires the Sprint 5a server to be running first.**

---

## 5. The Architecture — How It All Connects

### 5.1 The complete pipeline (XGBoost model → MCP server → agent)

This is the main "production" path — how a trained model ends up driving an autonomous agent:

```
   data/sbd_sales_data.csv ──────────────────────────────────────────────┐
   (raw weekly sales, source of truth)                                    │
            │                                                             │ (fallback
            ▼                                                             │  source if
   ┌─────────────────┐   trains    ┌──────────────────────┐              │  forecasts
   │  SPRINT 2       │ ──────────▶ │  XGBoost model        │              │  missing)
   │  Train XGBoost  │             │  models/xgb_..._model │              │
   │  (notebook)     │             │       .pkl            │              │
   └─────────────────┘             └──────────┬───────────┘              │
            │ logs/registers                  │                          │
            ▼                                  │ registered as            │
   ┌─────────────────┐                         │ models:/sbd_demand_      │
   │  SPRINT 3       │  mlflow.db  ◀────────────┤ forecast_xgb/latest      │
   │  MLflow registry│  (placeholder notebook; registry name used below)  │
   └─────────────────┘                         │                          │
                                               ▼                          │
                          ┌──────────────────────────────────────┐        │
                          │  generate_forecasts.py  (THE BRIDGE)  │        │
                          │  1. load model from MLflow            │        │
                          │     (── falls back to .pkl ──────────┐│        │
                          │  2. rebuild the 9 features            ││       │
                          │  3. roll a 4-week forecast per SKU    ││       │
                          │  4. compute weeks_of_supply + risk    ││       │
                          └───────────────────┬──────────────────┘│       │
                                              │ writes             │       │
                                              ▼                    │       │
                          ┌──────────────────────────────────────┐│       │
                          │  data/xgboost_forecasts.csv           ││       │
                          │  (real predictions: 80 SKU/warehouse) ││       │
                          └───────────────────┬──────────────────┘│       │
                                              │ read on startup     │       │
                                              ▼  (preferred)        │       │
                          ┌──────────────────────────────────────┐ │       │
                          │  sprint5_mcp_server.py  (FastAPI)     │◀┘       │
                          │  serves 4 tools at :8000/tools/...    │◀────────┘
                          │  XGBoost forecasts → tool responses   │  (fallback:
                          └───────────────────┬──────────────────┘   simple averages)
                                              │ HTTP (requests)
                              ┌───────────────┴────────────────┐
                              ▼                                ▼
                  ┌──────────────────────┐        ┌───────────────────────┐
                  │ sprint5_claude_mcp.py│        │  sprint6_agent.py      │
                  │ developer picks tools│        │  Claude picks tools    │
                  │ → Claude answers     │        │  in a loop → REPORT    │
                  └──────────────────────┘        └────────────────────────┘
```

So the data flows: **raw sales → trained XGBoost model → MLflow registry → `generate_forecasts.py` → `xgboost_forecasts.csv` → MCP server tools → Claude / the autonomous agent.** The bridge script is the join that lets the agent's weekly report be backed by real model predictions rather than simple averages.

### 5.2 Two ways the AI reaches the data

1. **RAG path (Sprint 4, standalone):** raw data → ChromaDB embeddings → semantic search → Claude. This path computes its own simple risk summary and does **not** use the XGBoost forecasts. Best for open-ended "find me things like X" questions.

```
   data/sbd_sales_data.csv ──▶ sprint4_rag.py ──▶ ChromaDB (./vector_db)
                                                        │ semantic search
                                  question ────────────▶│──▶ Claude ──▶ answer
```

2. **MCP / tool path (Sprints 5–6):** the complete pipeline above — model-backed forecasts served as FastAPI tools that Claude calls. Best for precise, structured lookups and for letting an agent take multi-step action.

**The shared backbone across both paths:** every part of the system speaks the same language — a `summary` table per SKU+warehouse with `avg_demand`, `weeks_of_supply`, and a `risk` label (HIGH < 16 weeks, MEDIUM < 20, else LOW). The only difference is whether the underlying demand number is a simple historical average or a real XGBoost forecast.

---

## 6. How to Run the Project

**Prerequisites**
- Install dependencies: `pip install -r requirements.txt`
- Create a `.env` file with your Claude key: `ANTHROPIC_API_KEY=sk-...`

**Run order**
1. **Sprint 1 & 2 notebooks** (in `notebooks/`) — run the cells top to bottom. Sprint 2 writes the model files into `models/`.
2. **Generate forecasts:** `python3 generate_forecasts.py` — loads the model and writes `data/xgboost_forecasts.csv`. Run this before the server so the server serves real XGBoost predictions. (Skip it and the server still works, just on simple averages.)
3. **Sprint 4** (RAG): `python sprint4_rag.py` — builds `vector_db/` and answers the demo questions.
4. **Sprint 5 server** — in one terminal: `uvicorn sprint5_mcp_server:app --reload --port 8000`
5. **Sprint 5 client** — in another terminal: `python sprint5_claude_mcp.py`
6. **Sprint 6 agent** — with the server still running: `python sprint6_agent.py`

> Sprints 5b and 6 both require the Sprint 5a server to be running on port 8000 first. Re-run step 2 whenever the model or data changes, then restart the server.

---

## 7. Glossary

- **SKU** — a unique product (e.g. `DCF899_ImpactDriver`). The unit of analysis is always SKU + warehouse.
- **Demand** — units sold in a week. This is the target the model predicts.
- **Weeks of supply** — current inventory ÷ average weekly demand. The core risk metric.
- **Stockout** — running out of inventory. The thing the system tries to prevent.
- **Lag feature** — a past value (e.g. demand 4 weeks ago) used as a model input.
- **MAE / MAPE** — accuracy metrics: average error in units / average error as a percentage.
- **XGBoost** — the gradient-boosted-trees algorithm used for forecasting.
- **MLflow** — tool for tracking and comparing model experiments.
- **LLM** — Large Language Model; here, **Claude** by Anthropic.
- **RAG** — Retrieval-Augmented Generation: fetch relevant data first, then let the LLM answer using it.
- **Embedding** — a numeric vector representing the meaning of text, enabling similarity search.
- **ChromaDB** — the vector database storing those embeddings.
- **MCP** — Model Context Protocol: a standard for giving an LLM safe, structured access to data via "tools."
- **Tool / tool-use** — a defined function the LLM can call (with named inputs and outputs).
- **Agent** — an LLM that autonomously chooses and chains tool calls in a loop to reach a goal.

---

## 8. Quick Reference — Files & Folders

| Path | What it is |
|------|-----------|
| `data/sbd_sales_data.csv` | Raw weekly sales data — the source of truth |
| `notebooks/sprint1_eda.ipynb` | Data exploration + charts |
| `notebooks/sprint2_model.ipynb` | XGBoost model training |
| `notebooks/sprint3_mlflow.ipynb` | MLflow tracking (empty placeholder) |
| `generate_forecasts.py` | **Bridge** — loads the XGBoost model (MLflow, pickle fallback), writes real 4-week forecasts |
| `data/xgboost_forecasts.csv` | Real model predictions consumed by the MCP server (produced by the bridge) |
| `sprint4_rag.py` | RAG pipeline (ChromaDB + Claude) |
| `sprint5_mcp_server.py` | FastAPI MCP server exposing 4 data tools |
| `sprint5_claude_mcp.py` | Claude using MCP tools (developer-orchestrated) |
| `sprint6_agent.py` | Autonomous agent (Claude-orchestrated) |
| `models/xgb_demand_model.pkl` | The trained model |
| `models/feature_list.txt` | The 9 model features |
| `models/model_results.txt` | Accuracy scores (baseline 21.1% → XGBoost 12.8% MAPE) |
| `outputs/*.png` | EDA and feature-importance charts |
| `vector_db/` | Persisted ChromaDB store (built by Sprint 4) |
| `mlflow.db`, `mlruns/` | MLflow tracking backend (scaffolding) |
| `requirements.txt` | Frozen Python environment |
| `.env` | Holds `ANTHROPIC_API_KEY` (not committed) |
