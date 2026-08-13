import os
import time
import json
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

# ---------------------------------------------------------
# 1. Configuration & Model Loading
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
POSSIBLE_PATHS = [
    "/kaggle/input/notebooks/matthewwisdom/agroai/qwen_farm_agent_gguf/qwen2.5-3b-instruct.Q4_K_M.gguf",
    str(BASE_DIR / "models" / "qwen2.5-3b-instruct.Q4_K_M.gguf"),
    "qwen_farm_agent.Q4_K_M.gguf"
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if not MODEL_PATH:
    MODEL_PATH = POSSIBLE_PATHS[1]
    print(f"Warning: Model file not found on disk. Tried: {POSSIBLE_PATHS}")

SYSTEM_PROMPT = """You are a localized farm management AI. You have access to the following tools: [{"name": "write_expenditure", "description": "Log a financial transaction into the expenditures table.", "parameters": ["category", "amount", "description"]}, {"name": "write_health_log", "description": "Record an animal's medical or physical event into the health_logs table.", "parameters": ["animal_id", "event_type", "notes"]}, {"name": "get_sensor_data", "description": "Retrieve readings from the telemetry_data table.", "parameters": ["node_id", "sensor_type"]}, {"name": "get_animal_record", "description": "Retrieve an animal's demographic and current status from the animals table.", "parameters": ["id"]}, {"name": "trigger_vision_reid", "description": "Execute the MegaDetector and MegaDescriptor pipeline on a locally saved image to identify an animal.", "parameters": ["image_filepath"]}, {"name": "query_knowledge_base", "description": "Search the RAG vector database for farming advice, disease treatment, or general agricultural knowledge.", "parameters": ["search_query"]}]"""

# ---------------------------------------------------------
# 2. Grammar Construction (JSON Schema AST)
# ---------------------------------------------------------
master_schema = {
    "anyOf": [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "enum": [
                            "write_expenditure",
                            "write_health_log",
                            "get_sensor_data",
                            "get_animal_record",
                            "trigger_vision_reid",
                            "query_knowledge_base"
                        ]
                    },
                    "arguments": {
                        "type": "object"
                    }
                },
                "required": ["function_name", "arguments"],
                "additionalProperties": False
            }
        },
        {
            "type": "string"
        }
    ]
}

print("Compiling JSON Schema Grammar...")
grammar = LlamaGrammar.from_json_schema(json.dumps(master_schema))

# ---------------------------------------------------------
# 3. Test Cases (Single & Multi-Turn Scenarios)
# ---------------------------------------------------------
test_scenarios = [
    {
        "name": "Single-Turn Hausa Sensor Query",
        "history": [
            {"role": "user", "content": "Duba min yanayin Danshi na Zone 2."}
        ]
    },
    {
        "name": "Single-Turn Pidgin Expenditure",
        "history": [
            {"role": "user", "content": "I buy 2 bags of goat feed for 45000 naira today."}
        ]
    },
    {
        "name": "Single-Turn Incomplete Log (Needs Clarification)",
        "history": [
            {"role": "user", "content": "Log say I buy medicine."}
        ]
    },
    {
        "name": "Multi-Turn Context Resolution",
        "history": [
            {"role": "user", "content": "Log say I buy medicine."},
            {"role": "assistant", "content": "Wetin be di amount wey you pay for di medicine, and which animal get am?"},
            {"role": "user", "content": "Na 12000 naira for di goat GT-004."}
        ]
    },
    {
        "name": "Guardrail Check (Non-Farm Query)",
        "history": [
            {"role": "user", "content": "Who win Premier League match yesterday?"}
        ]
    }
]

def format_chatml(system_prompt, messages):
    formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in messages:
        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted

# ---------------------------------------------------------
# 4. Benchmarking Execution
# ---------------------------------------------------------
def run_evaluation(llm, mode_name, use_grammar):
    print("\n" + "="*60)
    print(f"  RUNNING BENCHMARK: {mode_name.upper()}")
    print("="*60)

    for idx, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {idx}: {scenario['name']} ---")
        
        # Display context turns if multi-turn
        if len(scenario["history"]) > 1:
            print("History Context:")
            for msg in scenario["history"][:-1]:
                print(f"  [{msg['role'].upper()}]: {msg['content']}")
            print(f"Latest Turn:\n  [USER]: {scenario['history'][-1]['content']}")
        else:
            print(f"User > {scenario['history'][0]['content']}")

        prompt = format_chatml(SYSTEM_PROMPT, scenario["history"])
        
        start_time = time.time()
        output = llm(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>"],
            temperature=0.1,
            grammar=grammar if use_grammar else None
        )
        
        elapsed = time.time() - start_time
        response_text = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]

        print("Qwen > ", end="")
        if response_text.startswith("[") and "function_name" in response_text:
            try:
                tool_call = json.loads(response_text)
                print(f"[TOOL CALL]\n{json.dumps(tool_call, indent=2)}")
            except json.JSONDecodeError:
                print(f"[MALFORMED JSON] {response_text}")
        else:
            print(response_text)

        print(f"[Telemetry: {tokens} tokens in {elapsed:.2f}s | {tokens/elapsed:.1f} t/s]")

# ---------------------------------------------------------
# 5. Run Both Modes
# ---------------------------------------------------------
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Loading {MODEL_PATH} into memory...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        # Pass 1: Unconstrained Weights Evaluation
        run_evaluation(llm, "Without Grammar (Raw Weights)", use_grammar=False)
        
        # Pass 2: Schema Constrained Evaluation
        run_evaluation(llm, "With Grammar (JSON Schema Constrained)", use_grammar=True)
    else:
        print(f"Model file pending at {MODEL_PATH}. Place model file to run benchmark evaluation.")
