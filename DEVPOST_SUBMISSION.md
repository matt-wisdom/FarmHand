# DevPost Submission: FarmHand AI

**Africa Deep Tech Challenge 2026 (ADTC 2026): The Laptop LLM Challenge**  
**Submission Category:** Agriculture  

---

## 1. Project Overview

- **Project Title**: FarmHand AI: Edge-Native Multilingual Agricultural Intelligence
- **Tagline**: On-device LLM, Linear Programming feed optimizer, and epidemiological anomaly detection running 100% offline on commodity 8GB laptops.
- **Track**: Agriculture
- **GitHub Repository**: https://github.com/matt-wisdom/FarmHand
- **Hugging Face Model**: https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf
- **Demo Video URL**: [Link to 2-minute demonstration video]

---

## 2. DevPost Form Content

### Inspiration
Smallholder farmers produce more than 80% of agricultural output in Sub-Saharan Africa. In rural areas, producers face three major operational hurdles:
1. High commercial feed prices, which increased over 300% in local currency between 2024 and 2026.
2. Sudden livestock disease outbreaks (such as PPR, African Swine Fever, and Newcastle Disease) with limited access to veterinary extension staff.
3. The unreliability of cloud-based AI in remote locations due to recurring data costs, intermittent electrical grids, and limited cellular coverage.

FarmHand AI addresses this gap by running on standard refurbished laptops ($150 to $300) to deliver offline veterinary guidance, least-cost feed formulation, and flock accounting without recurring cloud fees or internet connectivity.

---

### What it Does
FarmHand AI is an offline, multi-farm management platform powered by a domain-adapted Small Language Model (`qwen_farm_agent`, fine-tuned on a high-fidelity synthetic multi-turn agricultural dataset) and domain-specific analytical engines:
- **Veterinary Guidance and Local RAG**: Answers livestock and crop health inquiries in **English, Nigerian Pidgin, and Hausa**, referencing 1,397 verified extension chunks from IITA, NAERLS, CGIAR, and FAO.
- **Linear Programming Feed Optimizer**: Calculates least-cost balanced animal rations across 22 local Nigerian ingredients in under 50 milliseconds, reducing feed costs by 25% to 40% compared to retail commercial feed.
- **Mortality Anomaly Surveillance**: Uses Scikit-Learn's `IsolationForest` and Median Absolute Deviation (MAD) statistics to flag disease outbreak spikes (>5% herd loss) and generate clinical triage summaries.
- **Relational Flock and Financial Ledger**: Provides deterministic inventory and expense tracking backed by SQLite.
- **Complete Offline Operation**: Operates with zero internet connectivity and consumes **~2.3 GB Peak RSS memory**, staying well below the 7.0 GB challenge limit.

---

### How We Built It
- **Domain-Adapted Language Model**: Fine-tuned Qwen 2.5 on a dedicated 4,000-sample multi-turn synthetic agricultural dataset (generated via Bedrock Mantle / `qwen.qwen3-235b-a22b-2507` teacher model covering single-turn JSON tool-calling, multi-turn parameter elicitation, veterinary RAG reasoning, and domain guardrails). Quantized to 4-bit Medium (`Q4_K_M` GGUF) and executed through `llama-cpp-python` with 2 physical CPU threads.
- **Hybrid Local Retrieval**: FastEmbed `bge-small-en-v1.5` dense embeddings (FAISS `IndexFlatIP`) combined with `bm25s` sparse lexical retrieval (60% BM25 / 40% dense).
- **Operations Research Solver**: Constrained Linear Programming optimization using `scipy.optimize.linprog` (HiGHS Interior Point solver).
- **Epidemiological Anomaly Detection**: Scikit-Learn `IsolationForest` paired with Median Absolute Deviation statistics.
- **Multilingual Support**: Helsinki-NLP MarianMT (`opus-mt-ha-en` / `opus-mt-en-ha`) with a dedicated Hausa agricultural glossary, alongside direct Nigerian Pidgin instruction-following.
- **Application Stack**: FastAPI backend, SQLite relational database, and an offline frontend with zero external CDN dependencies, clean SVG icons, and a flat documentation guide.

---

### Challenges Overcame
1. **Memory Containment Under 7 GB RAM on Pure CPU**: Running the language model, vector embeddings, BM25 indices, and statistical engines concurrently on CPU required strict memory limits, memory-mapped file loading, and thread pinning to avoid thermal throttling.
2. **Preventing Inventory Hallucinations**: We separated language generation from state tracking. Qwen parses and routes user requests to deterministic SQLite functions instead of tracking numbers in context memory.
3. **Natural Multilingual Prompting**: System prompts were written directly in natural Nigerian Pidgin and reinforced with a specialized Hausa terminology glossary to avoid unnatural translations.

---

### Accomplishments
- **Low Memory Footprint**: Achieved **~2.3 GB Peak RSS**, leaving over 65% memory headroom on standard 8GB laptops.
- **Measurable Feed Savings**: Formulated balanced rations yielding 35.8% cost savings per 50kg bag using accessible local ingredients.
- **Stable Offline Throughput**: Maintained generation speeds above 16 tokens per second on standard laptop CPUs.

---

### What We Learned
Small Language Models (SLMs) paired with targeted numerical solvers and statistical tools deliver faster, more reliable results for practical vertical workflows in resource-constrained environments than general-purpose cloud models.

---

### What's Next for FarmHand AI
- Adding support for Swahili, Yoruba, and Igbo agricultural vocabularies.
- Interfacing with low-cost local LoRa sensors for soil moisture and coop temperature monitoring.
- Cooperative field trials with smallholder farming clusters in Nigeria.

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
