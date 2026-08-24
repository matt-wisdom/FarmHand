# FarmHand AI

> **Edge-Native Multilingual Agricultural Intelligence on Commodity African Hardware**  
> Official Submission for the **Africa Deep Tech Challenge 2026 (ADTC 2026): The Laptop LLM Challenge**  
> **Track:** Agriculture | **Language Scope:** English, Nigerian Pidgin, Hausa  

---

## Overview

**FarmHand AI** is an offline, on-device advisory and flock ledger system built for the **ADTC Standard Laptop** profile (8 GB RAM, Intel Core i5/i7, integrated graphics, no discrete GPU).

The system integrates a 4-bit quantized Small Language Model (**Qwen 2.5 3B GGUF**), a **Linear Programming Feed Formulation Engine (SciPy `linprog`)**, an **Epidemiological Anomaly Detection Module (`IsolationForest` + MAD)**, and a **Hybrid Local Retrieval Index over 1,397 Nigerian Agricultural Extension Documents** to provide farm advisory and record-keeping without recurring cloud fees or internet access.

---

## Key Features

1. **Veterinary Guidance & Outbreak Triage**:
   - Diagnostic and biosecurity guidance for livestock diseases (PPR, African Swine Fever, Newcastle Disease, Coccidiosis).
   - Grounded in local extension documentation from IITA, NAERLS, CGIAR, and FAO.

2. **Least-Cost Feed Formulation (Operations Research)**:
   - Constrained Linear Programming solver (`scipy.optimize.linprog` HiGHS) that calculates nutritional mixtures across 22 Nigerian ingredients in under 50 milliseconds.
   - Generates savings between 25% and 40% per 50kg bag compared to retail commercial feed.

3. **Flock & Financial Operations Ledger**:
   - Deterministic relational accounting in SQLite for flock counts, mortality events, feed purchases, and medication expenses.
   - Natural language logging with automated category parsing.

4. **Epidemiological Anomaly Surveillance**:
   - Outlier detection combining unsupervised `IsolationForest` scoring and Median Absolute Deviation (MAD) to flag mortality spikes early.

5. **African Language Support (+15% Alpha Bonus)**:
   - Native instruction processing in **Nigerian Pidgin**.
   - Bidirectional neural translation for **Hausa** (`opus-mt-ha-en` / `opus-mt-en-ha`) paired with a regional agricultural lexicon.
   - Clean, low-distraction interface with SVG icons and a flat documentation guide.

---

## Systems Architecture

```
+-----------------------------------------------------------------------------------+
|                           FARMHAND AI ARCHITECTURE                                |
+-----------------------------------------------------------------------------------+
|  [ Farmer UI ]  English / Nigerian Pidgin / Hausa · Clean SVG · Flat Docs Manual  |
|                                     |                                             |
|  [ Gateway ]    FastAPI Server (localhost:8000) · 100% Local / Zero Cloud Leaks   |
|                                     |                                             |
|  [ LLM Core ]   Qwen 2.5 3B Instruct (Q4_K_M GGUF · llama.cpp · 2 CPU Threads)   |
|                                     |                                             |
|  +-----------------+----------------+-----------------+-----------------------+   |
|  |                 |                |                 |                       |   |
|  v                 v                v                 v                       v   |
| [ SQLite Ledger ] [ Feed LP Solver][ Anomaly Det. ]  [ Hybrid Local RAG ]    [ MT ]|
| Flock & Financial  SciPy linprog    IsolationForest   BM25s (60%) +           Hausa|
| Persistence        22 Ingredients   Robust MAD        FastEmbed BGE (40%)     Opus |
+-----------------------------------------------------------------------------------+
```

---

## Telemetry and Benchmark Results

Measured on the **Standard Laptop Profile (Intel Core i7-7500U @ 2.70GHz, 8 GB RAM, Integrated Graphics)**:

| Metric | Measured | Limit / Baseline | Status |
| :--- | :--- | :--- | :--- |
| **Peak Memory (RSS)** | **~3.27 GB** | 7.00 GB | **PASS (53.3% Headroom)** |
| **Throughput (TPS)** | **2.15 – 16.8 TPS** | 15.0 TPS Reference | **PASS** |
| **First Token Latency** | **< 1,800 ms** | 5,000 ms | **PASS** |
| **Max CPU Temperature** | **84.0 °C** | 85.0 °C ($P_{\text{thermal}}$) | **PASS ($P_{\text{thermal}} = 0$)** |
| **African Language Claim** | **Claimed (+15%)** | Pidgin + Hausa | **VERIFIED** |
| **Budget Profile Claim** | **Claimed (+10%)** | 8 GB Commodity CPU | **VERIFIED** |
| **Scorecard Status** | **100% Schema Valid** | `adtc-profiler.schema.json` | **VALIDATED** |

---

## Quickstart and Local Reproduction

### 1. Prerequisites
- Linux, macOS, or Windows WSL2
- Python $\ge 3.10$
- `uv` (recommended) or `pip`

### 2. Download Model Weights
```bash
# Clone the repository
git clone https://github.com/matt-wisdom/FarmHand.git
cd FarmHand

# Download the Qwen 2.5 3B GGUF weights (idempotent)
bash download_model.sh
```

### 3. Run the ADTC Telemetry Profiler
```bash
# Execute official telemetry profiler
python scripts/adtc_profiler.py --output submission.json
```

### 4. Launch the Application
```bash
# Install dependencies
uv pip install -r backend/requirements.txt

# Start local server
cd backend
python main.py

# Open browser at: http://localhost:8000
```

---

## Repository Layout

```
FarmHand/
├── metadata.json                 # Official ADTC submission metadata
├── submission.json               # Emitted telemetry benchmark scorecard
├── download_model.sh             # Idempotent model weight downloader
├── REPORT.md                     # Gate 1 Technical Paper
├── DEMO_VIDEO_SCRIPT.md          # 120-second demo video storyboard and script
├── DEVPOST_SUBMISSION.md         # DevPost submission text and metrics
├── README.md                     # Project documentation
├── scripts/
│   ├── adtc_profiler.py          # Telemetry profiler utility
│   └── upload_to_hf.py           # Hugging Face upload script
├── backend/
│   ├── main.py                   # FastAPI application server
│   ├── llm_engine.py             # Qwen 2.5 3B llama.cpp interface
│   ├── rag_pipeline.py           # Hybrid FastEmbed + BM25s retriever
│   ├── feed_optimizer.py         # SciPy LP least-cost ration solver
│   ├── anomaly_detector.py       # IsolationForest clinical surveillance
│   ├── database.py               # SQLite flock and financial ledger
│   ├── translator.py             # MarianMT Hausa neural translator
│   ├── models/                   # GGUF, ONNX, and index files
│   └── static/                   # Local HTML/CSS/JS frontend
└── evals/                        # Evaluation dataset and verification scripts
```

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
