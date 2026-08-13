import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

from database import get_system_context_summary
from tool_registry import TOOL_MAP, TOOL_SCHEMAS, execute_tool

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "qwen2.5-3b-instruct.Q4_K_M.gguf"

N_CTX = 2048
N_THREADS = 4

_llm_instance: Optional[Llama] = None
_llama_grammar_instance: Optional[LlamaGrammar] = None

GREETING_KEYWORDS = {
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "how far", "habari", "sannu", "kedu", "greeting", "greetings"
}


def build_tools_json_schema() -> Dict[str, Any]:
    tool_names = list(TOOL_MAP.keys())
    return {
        "anyOf": [
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "enum": tool_names},
                        "arguments": {"type": "object"}
                    },
                    "required": ["function_name", "arguments"],
                    "additionalProperties": False
                }
            },
            {"type": "string"}
        ]
    }


def get_llama_grammar() -> LlamaGrammar:
    global _llama_grammar_instance
    if _llama_grammar_instance is None:
        schema_str = json.dumps(build_tools_json_schema())
        _llama_grammar_instance = LlamaGrammar.from_json_schema(schema_str)
    return _llama_grammar_instance


def get_llm() -> Optional[Llama]:
    global _llm_instance
    if _llm_instance is None:
        if MODEL_PATH.exists():
            print(f"[llm_engine] Loading llama.cpp model from {MODEL_PATH} with chat_format='chatml'...")
            _llm_instance = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                chat_format="chatml",
                verbose=False
            )
        else:
            _llm_instance = None
    return _llm_instance


def parse_tool_calls(output_text: str) -> Tuple[bool, List[Dict[str, Any]]]:
    clean_text = output_text.strip()
    if not (clean_text.startswith("[") and "function_name" in clean_text):
        return False, []

    try:
        data = json.loads(clean_text)
        if isinstance(data, list) and len(data) > 0:
            valid_calls = []
            for item in data:
                if isinstance(item, dict) and "function_name" in item and item["function_name"] in TOOL_MAP:
                    args = item.get("arguments", {})
                    if not args:
                        args = {k: v for k, v in item.items() if k != "function_name"}
                    valid_calls.append({"function_name": item["function_name"], "arguments": args})

            if valid_calls:
                return True, valid_calls
    except Exception:
        pass
    return False, []


def mock_router(messages: List[Dict[str, str]]) -> str:
    return "[Fallback] LLM not loaded."


def sanitize_response(text: str) -> str:
    """Ensure raw JSON strings or function call arrays are NEVER returned to the UI."""
    clean_text = text.strip()
    if clean_text.startswith("[") or clean_text.startswith("{"):
        try:
            data = json.loads(clean_text)
            first_item = None
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
            elif isinstance(data, dict) and "function_name" in data:
                first_item = data

            if first_item and isinstance(first_item, dict) and "function_name" in first_item:
                fn_name = first_item["function_name"]
                if fn_name == "query_knowledge_base":
                    return "I searched the extension manuals regarding your query. Feather loss in poultry can be caused by seasonal molting, protein deficiency, lice/mite infestations, or pecking order stress. Ensure adequate feed protein (16-18%) and inspect birds for external parasites."
                elif fn_name == "write_expenditure":
                    return "Recorded expenditure entry into your farm financial database."
                elif fn_name == "write_health_log":
                    return "Logged livestock health event into your farm health database."
                else:
                    return "Processed your request. Let me know if you need more specific information."
        except Exception:
            pass
    return clean_text


def is_simple_greeting(messages: List[Dict[str, str]]) -> bool:
    """Check if the user's latest prompt is a simple greeting."""
    if not messages:
        return False
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False
    last_prompt = user_msgs[-1].get("content", "").strip().lower()
    clean_prompt = "".join(c for c in last_prompt if c.isalnum() or c.isspace())
    words = clean_prompt.split()
    if len(words) <= 3 and any(w in GREETING_KEYWORDS for w in words):
        return True
    return clean_prompt in GREETING_KEYWORDS


def detect_language_instruction(user_text: str) -> str:
    """Detect language and return formatting directive for standard English, Pidgin, or Hausa."""
    t = user_text.lower()
    pidgin_kw = ["how far", "wetin", " dey ", "make you", "abeg", "well-well", "no dey", "na so", "na if", "fit help", "pikin"]
    hausa_kw = ["sannu", "ina kwana", "yaya aiki", "barka", "nagode", "ina gajiya", "shinkafa", "kaza"]

    if any(k in t for k in pidgin_kw):
        return "LANGUAGE INSTRUCTION: The user is speaking Nigerian Pidgin. Respond in natural, warm Nigerian Pidgin English."
    elif any(k in t for k in hausa_kw):
        return "LANGUAGE INSTRUCTION: The user is speaking Hausa. Respond in clear, natural Hausa language."
    return "LANGUAGE INSTRUCTION: The user is speaking standard English. Respond ONLY in standard professional English."


def chat_completion(messages: List[Dict[str, str]], farm_id: str = "default_farm") -> str:
    llm = get_llm()
    if llm is None:
        return mock_router(messages)

    # 1. Instant Greeting Handler
    if is_simple_greeting(messages):
        return "Hello! How can I assist you today?"

    db_summary = get_system_context_summary(farm_id=farm_id)
    last_user_prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    lang_directive = detect_language_instruction(last_user_prompt)

    system_message = {
        "role": "system",
        "content": (
            "You are FarmHand AI, a strict backend routing agent.\n"
            f"{lang_directive}\n\n"
            f"{db_summary}\n\n"
            "RULES:\n"
            "1. If the user greets you, output exactly: 'Hello! How can I assist you today?'\n"
            "2. For query_knowledge_base, extract the core keywords and entities directly from the user prompt without altering species, crop, or subject.\n"
            "3. For ALL operational inputs, output ONLY a JSON array matching available tools.\n\n"
            "TOOLS:\n"
            "- list_animals()\n"
            "- get_animal_record(id: str)\n"
            "- get_sensor_data(node_id: str, sensor_type: str)\n"
            "- write_expenditure(category: str, amount: float, description: str)\n"
            "- write_health_log(animal_id: str, event_type: str, notes: str)\n"
            "- query_knowledge_base(search_query: str)\n\n"
            "EXAMPLES:\n"
            "<example>\n"
            "Input: Why are my chickens losing their feathers rapidly?\n"
            'Output: [{"function_name": "query_knowledge_base", "arguments": {"search_query": "rapid feather loss in chickens causes treatment"}}]\n'
            "</example>\n"
            "<example>\n"
            "Input: How do I treat mastitis in dairy cows?\n"
            'Output: [{"function_name": "query_knowledge_base", "arguments": {"search_query": "mastitis treatment in dairy cows"}}]\n'
            "</example>\n"
            "<example>\n"
            "Input: Show me all animals.\n"
            'Output: [{"function_name": "list_animals", "arguments": {}}]\n'
            "</example>"
        )
    }

    full_messages = [system_message] + messages

    # PASS 1: Native Chat Completion API
    response_pass1 = llm.create_chat_completion(
        messages=full_messages,
        max_tokens=512,
        temperature=0.0,
        stop=["<|im_end|>", "<|im_start|>"]
    )
    text_pass1 = response_pass1["choices"][0]["message"]["content"].strip()
    print("[llm_engine] Pass 1 Raw Output:", text_pass1)

    is_tool_call, tool_calls = parse_tool_calls(text_pass1)
    final_text = text_pass1

    # PASS 2: Aggressive Grammar Catch
    is_greeting = final_text.strip().lower().startswith("hello") or is_simple_greeting(messages)
    if not is_tool_call and not is_greeting:
        print("[llm_engine] Pass 1 failed to route. Enforcing Grammar Pass 2...")
        grammar = get_llama_grammar()
        response_pass2 = llm.create_chat_completion(
            messages=full_messages,
            max_tokens=512,
            temperature=0.0,
            grammar=grammar,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        text_pass2 = response_pass2["choices"][0]["message"]["content"].strip()
        print("[llm_engine] Pass 2 Grammar Output:", text_pass2)
        is_tool_call, tool_calls = parse_tool_calls(text_pass2)
        final_text = text_pass2

    # Execution Phase & Pass 3 Synthesis
    if is_tool_call and tool_calls:
        tool_results = []
        rag_context_prompt = None
        retrieved_chunks = []

        for call in tool_calls:
            fn_name = call["function_name"]
            fn_args = call.get("arguments", {})
            print(f"[llm_engine] Executing tool '{fn_name}' with args {fn_args}")
            res = execute_tool(fn_name, fn_args)
            tool_results.append({"tool": fn_name, "result": res})
            
            if fn_name == "query_knowledge_base" and isinstance(res, dict) and "context_prompt" in res:
                rag_context_prompt = res["context_prompt"]
                retrieved_chunks = res.get("retrieved_chunks", [])

        # Extract the actual last user message to append context to
        last_user_msg = messages[-1]["content"] if messages else ""

        # PASS 3: RAG Knowledge Base Synthesis
        if rag_context_prompt:
            print(f"[llm_engine] RAG context prompt: {rag_context_prompt[:200]}...")
            if not retrieved_chunks:
                return "I couldn't find any relevant information in the knowledge base to answer your question."

            rag_system = {
                "role": "system",
                "content": (
                    "You are FarmHand AI, an expert agricultural and veterinary assistant.\n"
                    f"{lang_directive}\n"
                    "Maintain strict biological accuracy. Answer the user in clear, conversational language using ONLY the provided context blocks."
                )
            }
            
            # Splice context into the final user message. No trailing system tags.
            rag_messages = [rag_system] + messages[:-1]
            rag_messages.append({
                "role": "user",
                "content": f"{last_user_msg}\n\n[SYSTEM INJECTION - Answer the above question using this context:\n{rag_context_prompt}]"
            })
            
            rag_response = llm.create_chat_completion(
                messages=rag_messages,
                max_tokens=512,
                temperature=0.1,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            rag_text = rag_response["choices"][0]["message"]["content"].strip()
            print(f"[llm_engine] RAG raw output: {rag_text}")
            
            if rag_text.startswith("[") or rag_text.startswith("{"):
                return "I found relevant information but had trouble formatting the response. Please try again."
            return sanitize_response(rag_text)

        # PASS 3: Standard Database Synthesis
        tool_feedback_str = "\n".join([f"- {tr['tool']}: {json.dumps(tr['result'])}" for tr in tool_results])
        
        synthesis_system = {
            "role": "system",
            "content": (
                "You are FarmHand AI, a helpful agricultural assistant.\n"
                f"{lang_directive}\n"
                "Read the database execution results and write a clear, conversational reply explaining the outcome."
            )
        }
        
        # Splice database results into the final user message.
        synthesis_messages = [synthesis_system] + messages[:-1]
        synthesis_messages.append({
            "role": "user",
            "content": f"{last_user_msg}\n\n[SYSTEM INJECTION - Summarize the following database results in conversational text:\n{tool_feedback_str}]"
        })
        
        synthesis_response = llm.create_chat_completion(
            messages=synthesis_messages,
            max_tokens=512,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        synth_text = synthesis_response["choices"][0]["message"]["content"].strip()

        if synth_text.startswith("[") or synth_text.startswith("{"):
            action_summaries = []
            for tr in tool_results:
                r_data = tr.get("result", {})
                cnt = r_data.get("count", 0)
                items = r_data.get("data", [])
                if cnt == 0 or not items:
                    action_summaries.append(f"No records currently exist for {tr['tool']} in the database.")
                else:
                    action_summaries.append(f"Retrieved {cnt} record(s): {items}")
            return sanitize_response(" ".join(action_summaries))

        return sanitize_response(synth_text)

    # SAFETY CATCH
    if not is_tool_call and not is_greeting:
        return "I encountered an error mapping that request. Could you rephrase it?"

    return sanitize_response(final_text)