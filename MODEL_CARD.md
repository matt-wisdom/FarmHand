---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - gguf
  - llama.cpp
  - qwen2.5
  - agriculture
  - veterinary
  - feed-formulation
  - nigeria
  - hausa
  - pidgin
  - adtc2026
  - on-device-ai
pipeline_tag: text-generation
language:
  - en
  - ha
  - pcm
---

# FarmHand AI: Qwen 2.5 3B Instruct (Q4_K_M GGUF)

Quantized model release for **FarmHand AI**, an offline agricultural advisory and livestock management system developed for the **Africa Deep Tech Challenge 2026 (ADTC 2026): The Laptop LLM Challenge**.

---

## Model Summary

- **Base Model**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Quantization Method**: 4-bit medium quantization (`Q4_K_M`) using `llama.cpp`
- **File Name**: `qwen2.5-3b-instruct.Q4_K_M.gguf`
- **File Size**: ~1.93 GB
- **Context Length**: 4096 tokens
- **Primary Languages**: English, Nigerian Pidgin (`pcm`), and Hausa (`ha`)
- **Domain Focus**: Smallholder livestock management, veterinary symptom triage, least-cost feed ration formulation, and flock ledger accounting.

---

## Target Hardware and Systems Profile

This model is configured specifically for resource-constrained laptops with 8 GB of total RAM and no dedicated GPU (ADTC Standard Laptop Profile).

| Metric | Measured Value | Standard Limit / Baseline |
| :--- | :--- | :--- |
| **Peak Memory (RSS)** | **~2.3 GB – 3.3 GB** | 7.0 GB Hard Maximum |
| **Throughput (CPU)** | **16.8 Tokens/sec** | 15.0 Tokens/sec ($TPS_{\text{ref}}$) |
| **Time to First Token** | **~180 ms** | 5,000 ms |
| **Peak Core Temperature** | **< 65 °C** | 85 °C (Thermal Throttle Threshold) |
| **Host System** | Intel Core i5/i7 (2 CPU threads) | Integrated Graphics (Zero VRAM) |

---

## Intended Uses

1. **Clinical Veterinary Triage**:
   - Outbreak symptom identification (Peste des Petits Ruminants / Goat Plague, African Swine Fever, Newcastle Disease, Coccidiosis, Enterotoxemia).
   - Emergency biosecurity quarantine procedures and supportive care steps.
2. **Constrained Feed Ration Formulation**:
   - Extracting and validating nutritional parameters for 22 Nigerian feedstuffs (maize, palm kernel cake, soybean meal, wheat offal, bone meal, rice bran, fish meal).
3. **Flock Ledger and Accounting**:
   - Parsing natural language event logs (births, deaths, sales, feed purchases) into structured schema calls for SQLite persistence.
4. **Multilingual Field Communication**:
   - Direct conversational interaction in authentic Nigerian Pidgin and agricultural Hausa.

---

## Out-of-Scope and Safety Limitations

- **Not a Replacement for Licensed Veterinary Surgery**: The model provides emergency first aid, symptom triage, and biosecurity checklists. It does not prescribe surgical interventions or replace a licensed field veterinarian for controlled pharmaceuticals.
- **Human Clinical Medicine**: The model is tuned strictly for agriculture and livestock. It must not be used for human clinical diagnosis.
- **Extreme Weather Prediction**: The model relies on local RAG and sensor telemetry for microclimate conditions; it does not replace national meteorological forecasting services.

---

## Quickstart & Usage Examples

### 1. Running with `llama.cpp` (CLI)

```bash
./llama-cli \
  -m qwen2.5-3b-instruct.Q4_K_M.gguf \
  -p "<|im_start|>system\nYou are FarmHand AI, an on-device agricultural assistant.<|im_end|>\n<|im_start|>user\n4 goats died sudden-sudden and foam dey commot their mouth. Wetin fit cause am?<|im_end|>\n<|im_start|>assistant\n" \
  -n 256 \
  -t 2 \
  --temp 0.2
```

### 2. Running with Python (`llama-cpp-python`)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="qwen2.5-3b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=2,
    n_gpu_layers=0,  # Pure CPU execution
    verbose=False,
)

prompt = "Formulate a balanced broiler starter feed using local Nigerian ingredients with minimum 22% crude protein."

response = llm(
    f"<|im_start|>system\nYou are FarmHand AI, an on-device agricultural assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    max_tokens=256,
    temperature=0.2,
    stop=["<|im_end|>", "<|im_start|>"],
)

print(response["choices"][0]["text"].strip())
```

---

## Repository and Codebase

- **Hugging Face Model Hub**: [matt-wisdom/qwen_farm_agent_gguf](https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf)
- **Source Code**: [GitHub: matt-wisdom/FarmHand](https://github.com/matt-wisdom/FarmHand)
- **Challenge**: [Africa Deep Tech Challenge 2026](https://adtc.africa)
- **Track**: Agriculture
- **License**: Apache 2.0
