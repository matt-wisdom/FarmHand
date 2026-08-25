# DevPost Submission: FarmHand AI

**Africa Deep Tech Challenge 2026 (ADTC 2026): The Laptop LLM Challenge**  
**Submission Category:** Agriculture  

---

## 1. Project Overview

- **Project Title**: FarmHand: Edge-Native Multilingual Agricultural Intelligence
- **Tagline**: Local, offline AI assistant that tracks farm records, calculates cheap balanced feed mixtures, and diagnoses livestock illnesses in English, Hausa, and Pidgin directly on basic laptop hardware.
- **Track**: Agriculture
- **GitHub Repository**: https://github.com/matt-wisdom/FarmHand
- **Hugging Face Model**: https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf

---

## 2. DevPost Form Content

### Inspiration
Growing up, my mom ran a goat and chicken farm. Everything on the farm was tracked manually on paper notebooks or kept in memory: flock headcounts, feed bags bought, market sales, and disease outbreaks.

When paper notebooks got wet, misplaced, or torn, we lost our records. It was impossible to know our actual profit margins or calculate true mortality rates. When animals fell sick, veterinary doctors were either hours away or too expensive, so diagnosis was largely trial and error. Mixing feed was another challenge: commercial feed was expensive, and trying to formulate our own rations from local market grains meant guessing proportions and risking animal malnutrition.

Most modern agricultural software relies on cloud servers, paid monthly subscriptions, and reliable high-speed internet. That does not work in rural farming communities where internet is unstable and cellular data is a luxury. I built FarmHand to solve the problems we faced on our farm: a local, offline AI assistant that tracks farm records, calculates cheap, balanced feed mixtures, and diagnoses livestock illnesses in English, Hausa, and Pidgin directly on basic laptop hardware without an internet connection.

---

### What it Does
FarmHand is an offline, multi-farm management assistant and veterinary diagnostic tool designed for smallholders and livestock farmers:
- **Veterinary & Clinical Triage**: Diagnoses diseases and symptoms (e.g., PPR, Coccidiosis, Newcastle disease, Colibacillosis, Enterotoxemia) across poultry, goats, cattle, sheep, pigs, and catfish, providing immediate emergency care and biosecurity steps.
- **Least-Cost Feed Optimizer**: Uses linear programming to calculate nutritionally balanced feed recipes for broilers, layers, catfish, and ruminants using local, low-cost ingredients (maize, roasted soybean meal, fish meal, rice bran, limestone, bone meal).
- **Flock & Financial Ledgers**: Records livestock counts, births, sales, and mortalities, and logs farm operational expenses by category.
- **Anomaly Detection**: Monitors mortality spikes and feed consumption trends across historical records, alerting farmers to emerging outbreaks.
- **Semantic Farm Memory**: Stores persistent notes about farm infrastructure, water sources, and housing setups, recalling them during relevant advisory queries.
- **Multi-Language Interaction**: Communicates fluently in Standard English, Hausa, and Nigerian Pidgin.

---

### How We Built It
FarmHand runs completely on-device with zero external API calls or telemetry:

```
  +-----------------------------------------------------------------------+
  |                             Farmer Input                              |
  |                      (English / Hausa / Pidgin)                       |
  +-----------------------------------+-----------------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  |  1. Pre-Processing & Fast Path                                        |
  |  - Instant greeting interceptor (0.00s latency)                       |
  |  - Offline MarianMT translation (Hausa -> English)                    |
  +-----------------------------------+-----------------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  |  2. Tool Router (Qwen 2.5 3B Q4_K_M via llama.cpp)                    |
  +----------+------------------------+-------------------------+---------+
             |                        |                         |
             v                        v                         v
  +--------------------+   +---------------------+   +--------------------+
  |  Veterinary RAG    |   | Linear Programming  |   | SQLite Ledgers     |
  |  - FAISS Vector    |   | Feed Optimizer      |   | - Flock Headcounts |
  |  - BM25 Okapi      |   | - SciPy Simplex     |   | - Expense Ledger   |
  |  - 24k Vet Chunks  |   | - Nutrient Specs    |   | - Farm Memories    |
  +----------+---------+   +----------+----------+   +----------+---------+
             |                        |                         |
             +------------------------+-------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  |  3. Context-Conditioned Synthesis & SSE Streaming                     |
  |  - Anti-JSON token biasing                                            |
  |  - Real-time token streaming to browser via SSE                       |
  |  - Offline MarianMT translation (English -> Hausa)                    |
  +-----------------------------------+-----------------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  |                             Web Frontend                              |
  |                 (Single-file Vue 3 + CSS Interface)                   |
  +-----------------------------------------------------------------------+
```

#### 1. Offline Language Model Runtime
- **Model**: Qwen 2.5 3B Instruct quantized to 4-bit (`qwen2.5-3b-instruct.Q4_K_M.gguf`, ~1.9 GB), domain-adapted on a 4,000-sample multi-turn synthetic agricultural dataset.
- **Execution**: `llama.cpp` using AVX2 CPU SIMD instructions with multi-threaded execution scaled to CPU core count ($n_{\text{threads}} = 2\text{--}4$, $n_{\text{batch}} = 512$).
- **Streaming**: FastAPI `StreamingResponse` sends Server-Sent Events (SSE) to render words progressively in the browser.

#### 2. Hybrid Veterinary RAG Pipeline
The knowledge base contains 24,393 reference document chunks extracted from tropical agricultural and veterinary manuals:
- **Dense Vector Search**: Sentence-Transformers `all-MiniLM-L6-v2` / FastEmbed embeddings in FAISS (`IndexFlatIP`).
- **Sparse Keyword Search**: BM25 Okapi index over tokenized chunks.
- **Reciprocal Rank Fusion (RRF)**: Combines dense and sparse results:
  $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $k = 60$ and $r_m(d)$ is the rank of chunk $d$ in search system $m$.
- **Relevance Gate**: Chunks scoring below similarity floor ($S_{\min} = 0.55$) are filtered out to prevent inaccurate synthesis.

#### 3. Least-Cost Feed Formulation via Linear Programming
Rather than relying on language model arithmetic, feed formulation is handled deterministically with `scipy.optimize.linprog`:
$$\min_{x} \text{Cost} = \sum_{i=1}^{n} c_i x_i$$
Subject to nutritional requirements and batch limits:
$$\sum_{i=1}^{n} A_{ji} x_i \ge b_j \quad (\text{Crude Protein, Energy, Calcium, Phosphorus})$$
$$\sum_{i=1}^{n} x_i = W_{\text{batch}} \quad (\text{Total Batch Weight})$$
$$l_i \le x_i \le u_i \quad (\text{Minimum / Maximum Ingredient Bounds})$$
Where $c_i$ is the cost per kilogram of ingredient $i$, and $x_i$ is the allocated weight in kilograms.

#### 4. Local Persistence & Frontend
- **Database Architecture**: SQLite strictly isolates static reference document chunks (`knowledge_chunks.db`) from user farm records and chat history (`farm_local.db`).
- **Frontend**: Lightweight single-file Vue 3 interface with dark/light mode persistence and zero external CDN requirements.

---

### Challenges We Ran Into

1. **Grammar Parsing Overhead on Large Vocabularies**:
   Qwen 2.5 uses a 152,064-token vocabulary. Evaluating GBNF grammar masks over 152k logits on CPU added substantial per-token latency. We solved this by using targeted logit-biasing on JSON control tokens and setting strict token budgets ($N_{\max} = 128$) on the routing pass.
2. **Memory Contention on Consumer Hardware**:
   Running a 3B LLM, FAISS indices, translation models, and an HTTP server concurrently on 8 GB RAM caused OS swap thrashing. We addressed this through lazy model loading, KV cache budgeting ($N_{\text{ctx}} = 2048 / 4096$), and pre-calculating index lookups.
3. **Translation Corpus Bias**:
   Generic MarianMT models trained on religious texts introduced Biblical phrasing into Hausa agricultural output. We added a post-processing filter that catches religious artifacts and falls back to clean technical English when translation divergence is detected.
4. **Data Isolation**:
   Bundling reference knowledge with active user farm logs previously risked pushing private farm ledgers to version control. We separated the storage into a read-only static knowledge database and an untracked local farm database.

---

### Accomplishments That We're Proud Of

- **True 100% Offline Autonomy**: The entire pipeline (LLM inference, vector search, linear programming, neural translation, and ledger database) runs without sending a single byte over the internet.
- **Sub-50ms Feed Formulation**: Solves complex linear programming equations for balanced broiler, layer, catfish, and ruminant rations in under 50 milliseconds.
- **Responsive Edge Inference**: Delivers time-to-first-token in under 2 seconds on a standard dual-core laptop CPU using multi-threaded AVX2 instructions and SSE streaming.
- **Dialect Inclusivity**: Provides genuine support for Nigerian Pidgin and Hausa, meeting rural farmers in the languages they speak daily.

---

### What We Learned

- **Deterministic Solvers + LLM Router > Pure Generative Models**: LLMs excel at parsing informal language and user intent, while mathematical algorithms (like linear programming) should always handle nutritional arithmetic.
- **Fast-Path Interceptors Save Real Compute**: Greeting filters and fast intent handlers eliminate unnecessary model evaluations for common pleasantries, dropping latency from seconds to 0.00 s.
- **Simplicity Over Abstraction**: Writing modular, single-purpose functions without unnecessary layers makes edge AI easier to debug, test, and run on constrained hardware.

---

### What's Next for FarmHand

- **On-Device Voice Interface**: Integrating lightweight offline speech-to-text (Whisper) and text-to-speech models so non-literate farmers can speak directly to FarmHand.
- **Computer Vision Triage**: Adding quantized vision models for on-device visual disease diagnosis from photos of poultry droppings, skin lesions, and crop leaves.
- **Local Mesh Synchronization**: Enabling multi-device data syncing over local Wi-Fi or Bluetooth mesh networks so farm workers can update flock records in the field without internet access.

---

## 3. Self-Reported Profiler Metrics

| Field on DevPost | Value |
| :--- | :--- |
| **Performance Score ($S_{perf}$)** | **100.00** (Throughput: 16.82 TPS vs 15.0 TPS Reference) |
| **Efficiency Score ($S_{eff}$)** | **67.29** (Peak RAM: 2.29 GB / 7.0 GB limit) |
| **Accuracy Score ($S_{acc}$)** | **92.00** |
| **African Language Alpha Bonus Claimed?** | **Yes (+15%)** (Nigerian Pidgin + Hausa MarianMT) |
| **Budget Laptop Profile Claimed?** | **Yes (+10%)** (Runs on 8GB commodity CPU hardware) |
| **Projected Composite Score** | **113.17 Points** |
