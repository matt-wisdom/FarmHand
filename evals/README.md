# FarmHand AI Evaluation & Benchmarking Suite

An automated evaluation framework for benchmarking the localized FarmHand AI model across veterinary clinical triage, livestock & aquaculture feeding formulations, crop agronomy, multi-turn diagnostics, telemetry tool calling, and bilingual language fidelity.

---

## Directory Structure

```
evals/
├── eval_dataset.json     # Primary benchmark dataset containing all evaluation specs
├── run_evals.py          # Benchmark runner engine with report generation
├── add_eval.py           # CLI & Interactive wizard for adding new evaluations
├── reports/              # Automatically generated Markdown & JSON benchmark reports
└── README.md             # Documentation and usage guide
```

---

## 1. Running Benchmarks

### Run all benchmarks
```bash
python evals/run_evals.py
```

### Filter by Category
```bash
python evals/run_evals.py --category veterinary_diagnosis
python evals/run_evals.py --category feed_formulation
python evals/run_evals.py --category tool_calling
```

### Filter by Language Mode
```bash
# Benchmark only English-mode queries
python evals/run_evals.py --language english

# Benchmark only Nigerian Pidgin queries
python evals/run_evals.py --language pidgin
```

### Run a Single Specific Benchmark
```bash
python evals/run_evals.py --id eval_vet_01_goat_tetanus_en
```

### Verbose Mode (Displays full outputs)
```bash
python evals/run_evals.py --verbose
```

---

## 2. Adding New Evaluations

### Option A: Interactive Wizard
Run the wizard and follow the interactive prompts:
```bash
python evals/add_eval.py
```

### Option B: CLI Flags
```bash
python evals/add_eval.py \
  --id eval_vet_swine_fever \
  --category veterinary_diagnosis \
  --language english \
  --query "My pigs have high fever, red skin blotches, and are dying rapidly. What could be the cause?" \
  --must-contain "african swine fever,virus,biosecurity" \
  --description "African Swine Fever symptom diagnosis"
```

### Option C: Direct JSON Editing
Open `evals/eval_dataset.json` and add a new entry to the `"evals"` array:
```json
{
  "id": "eval_feed_piglet_starter_en",
  "category": "feed_formulation",
  "language": "english",
  "description": "Piglet creep feed formulation with local grains",
  "farm_id": "farm_9eb3f441",
  "messages": [
    {
      "role": "user",
      "content": "How do I formulate creep feed for weaned piglets using maize and fish meal?"
    }
  ],
  "expected_tools": ["query_knowledge_base"],
  "must_contain": ["protein", "maize", "digest"],
  "must_not_contain": ["dey", "wey", "wetin", "una", "dem", " di ", "well-well"]
}
```

---

## 3. Evaluation Assertions & Checks

Each evaluation test case validates:
1. **Tool Invocation Accuracy**: Verifies Pass 1 selected the proper tool (e.g. `query_knowledge_base`, `get_sensor_data_node`, `write_health_log`, `write_expenditure`).
2. **Positive Assertions (`must_contain`)**: Verifies critical domain knowledge and clinical terms are present in the response.
3. **Language Fidelity (`must_not_contain`)**:
   - In **English mode**, strictly validates that no Nigerian Pidgin particles (`dey`, `wey`, `wetin`, `una`, `dem`, `di`, `am`, etc.) appear.
   - In **Pidgin mode**, validates natural Nigerian Pidgin tone.
4. **Structural Formatting**: Ensures definitions and conceptual questions produce informative prose paragraphs rather than forced numbered lists.
5. **No Hallucinated JSON**: Guarantees raw internal JSON dictionaries or function call syntax are never leaked to the user.
6. **Latency Tracking**: Records TTFT and total latency per query for performance profiling.
