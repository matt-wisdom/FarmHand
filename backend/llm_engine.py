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

N_CTX = 4096  
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
    clean_text = text.strip()
    if clean_text.startswith("[") or clean_text.startswith("{"):
        return "I processed your request, but encountered an error formatting the final output."
    return clean_text


def is_simple_greeting(messages: List[Dict[str, str]]) -> bool:
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
    t = user_text.lower()
    pidgin_kw = ["how far", "wetin", " dey ", "make you", "abeg", "well-well", "no dey", "na so", "na if", "fit help", "pikin"]
    hausa_kw = ["sannu", "ina kwana", "yaya aiki", "barka", "nagode", "ina gajiya", "shinkafa", "kaza"]

    if any(k in t for k in pidgin_kw):
        return "The user is speaking Nigerian Pidgin. Respond in natural, warm Nigerian Pidgin English."
    elif any(k in t for k in hausa_kw):
        return "The user is speaking Hausa. Respond in clear, natural Hausa language."
    return "The user is speaking standard English. Respond ONLY in standard professional English."


def generate_stateless_answer(llm: Llama, context_data: str, user_question: str, lang_directive: str, db_summary: str) -> str:
    """Pass 3: Forcing English generation using Assistant Prefilling on the raw completion API."""
    
    # We manually construct the ChatML string to force the model's opening words.
    raw_prompt = (
        "<|im_start|>system\n"
        "You are FarmHand AI, an expert agricultural and veterinary assistant.\n"
        f"{lang_directive}\n\n"
        f"FARM CONTEXT:\n{db_summary}\n\n"
        "Write a clear, conversational response to the user based ONLY on the provided Reference Material. "
        "Do not output JSON.<|im_end|>\n"
        "<|im_start|>user\n"
        f"User's Request: {user_question}\n\n"
        f"Reference Material:\n{context_data}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Based on the provided information, "
    )
    
    # Use raw create_completion to bypass chat wrapper formatting
    response = llm.create_completion(
        prompt=raw_prompt,
        max_tokens=512,
        temperature=0.1,
        stop=["<|im_end|>", "<|im_start|>"]
    )
    
    # Stitch our forced prefix back together with the model's generated text
    generated_text = response["choices"][0]["text"].strip()
    return f"Based on the provided information, {generated_text}"


def chat_completion(messages: List[Dict[str, str]], farm_id: str = "default_farm", thread_id: Optional[str] = None, language: str = "auto") -> str:
    llm = get_llm()
    if llm is None:
        return mock_router(messages)

    if is_simple_greeting(messages):
        return "Hello! I am FarmHand AI. How can I assist you with your farm today?"

    db_summary = get_system_context_summary(farm_id=farm_id)
    last_user_prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")

    # Use explicit language selection or auto-detect
    if language and language != "auto":
        if language == "english":
            lang_directive = "The user expects English. Respond in clear, professional English."
        elif language == "hausa":
            lang_directive = "The user expects Hausa. Respond in clear Hausa language."
        elif language == "pidgin":
            lang_directive = "The user expects Nigerian Pidgin. Respond in warm Nigerian Pidgin English."
        else:
            lang_directive = detect_language_instruction(last_user_prompt)
    else:
        lang_directive = detect_language_instruction(last_user_prompt)

    system_message = {
        "role": "system",
        "content": (
            "You are the internal JSON routing engine for FarmHand AI.\n"
            "Your ONLY job is to output a JSON array of tool calls based on the user's request.\n"
            "TOOLS:\n"
            "- list_animals()\n"
            "- get_animal_record(id: str)\n"
            "- get_sensor_data(node_id: str, sensor_type: str)\n"
            "- write_expenditure(category: str, amount: float, description: str)\n"
            "- write_health_log(animal_id: str, event_type: str, notes: str)\n"
            "- query_knowledge_base(search_query: str)"
        )
    }

    full_messages = [system_message] + messages
    grammar = get_llama_grammar()

    # PASS 1: Native API with FORCED GRAMMAR
    print(f"[llm_engine] Running Pass 1 for thread_id {thread_id} with strict LlamaGrammar enforcement...")
    response_pass1 = llm.create_chat_completion(
        messages=full_messages,
        max_tokens=512,
        temperature=0.0,
        grammar=grammar,
        stop=["<|im_end|>", "<|im_start|>"]
    )
    text_pass1 = response_pass1["choices"][0]["message"]["content"].strip()
    print("[llm_engine] Pass 1 Raw Output:", text_pass1)

    is_tool_call, tool_calls = parse_tool_calls(text_pass1)

    if not is_tool_call:
        return "I encountered an error understanding that request. Could you please rephrase it?"

    # Execution Phase
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

    # PASS 3: RAG Synthesis (using raw completion prefilling)
    if rag_context_prompt:
        if not retrieved_chunks:
            return "I couldn't find any relevant information in the knowledge base to answer your question."
        
        rag_text = generate_stateless_answer(llm, rag_context_prompt, last_user_prompt, lang_directive, db_summary)
        print(f"[llm_engine] RAG stateless output:\n{rag_text}")
        return sanitize_response(rag_text)

    # PASS 3: Database Synthesis (using raw completion prefilling)
    tool_feedback_str = "\n".join([f"- {tr['tool']}: {json.dumps(tr['result'])}" for tr in tool_results])
    synth_text = generate_stateless_answer(llm, tool_feedback_str, last_user_prompt, lang_directive, db_summary)
    
    return sanitize_response(synth_text)