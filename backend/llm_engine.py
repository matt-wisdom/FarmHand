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


# Unified response schema for ALL Pass 1 outputs
def build_unified_response_schema() -> Dict[str, Any]:
    tool_names = list(TOOL_MAP.keys())
    return {
        "type": "object",
        "oneOf": [
            {
                # Tool call format
                "properties": {
                    "action": {"const": "tool_call"},
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "function_name": {"type": "string", "enum": tool_names},
                                "arguments": {"type": "object"}
                            },
                            "required": ["function_name", "arguments"]
                        }
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["action", "data"]
            },
            {
                # Direct response format
                "properties": {
                    "action": {"const": "respond"},
                    "text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["action", "text"]
            },
            {
                # Error format
                "properties": {
                    "action": {"const": "error"},
                    "message": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["action", "message"]
            }
        ]
    }


def get_llama_grammar() -> LlamaGrammar:
    global _llama_grammar_instance
    if _llama_grammar_instance is None:
        schema_str = json.dumps(build_tools_json_schema())
        _llama_grammar_instance = LlamaGrammar.from_json_schema(schema_str)
    return _llama_grammar_instance


_unified_grammar_instance = None


def get_unified_grammar() -> LlamaGrammar:
    """Grammar for unified JSON response format (all outputs)."""
    global _unified_grammar_instance
    if _unified_grammar_instance is None:
        schema_str = json.dumps(build_unified_response_schema())
        _unified_grammar_instance = LlamaGrammar.from_json_schema(schema_str)
    return _unified_grammar_instance


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
    """Parse both unified JSON format and legacy tool call format."""
    clean_text = output_text.strip()

    # Try unified format: {"action": "tool_call", "data": [...]} or {"action": "respond", "text": "..."}
    if clean_text.startswith("{"):
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict):
                action = data.get("action")

                if action == "tool_call":
                    tool_calls = []
                    for item in data.get("data", []):
                        if isinstance(item, dict) and "function_name" in item:
                            fn_name = item["function_name"]
                            if fn_name in TOOL_MAP:
                                args = item.get("arguments", {})
                                tool_calls.append({"function_name": fn_name, "arguments": args})
                    if tool_calls:
                        return True, tool_calls
                    return False, []

                elif action == "respond":
                    # Direct response - return special marker
                    return False, [{"_direct": True, "text": data.get("text", ""), "confidence": data.get("confidence", 0.9)}]

                elif action == "error":
                    return False, [{"_error": True, "message": data.get("message", "Unknown error")}]
        except Exception:
            pass

    # Try legacy format: [{"function_name": ...}]
    if clean_text.startswith("[") and "function_name" in clean_text:
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


def get_text_grammar() -> LlamaGrammar:
    """Grammar that forces plain text - no JSON, no brackets."""
    grammar_text = r'''
    text ::= [a-zA-Z0-9 .,!?\'"-]+
    '''
    return LlamaGrammar.from_json_schema('{"type": "string"}')


_text_grammar_instance = None


def get_text_only_grammar() -> LlamaGrammar:
    global _text_grammar_instance
    if _text_grammar_instance is None:
        _text_grammar_instance = get_text_grammar()
    return _text_grammar_instance


def generate_stateless_answer(llm: Llama, context_data: str, user_question: str, lang_directive: str) -> str:
    """Pass 3: Pure data extraction. Use grammar to force natural language."""
    system_prompt = {
        "role": "system",
        "content": (
            "You are FarmHand AI. Answer the farmer's question in plain English.\n"
            f"{lang_directive}\n\n"
            "Write 1-3 sentences. Do not use brackets or code."
        )
    }

    user_prompt = {
        "role": "user",
        "content": f"Context: {context_data}\n\nQuestion: {user_question}\n\nAnswer:"
    }

    response = llm.create_chat_completion(
        messages=[system_prompt, user_prompt],
        max_tokens=384,
        temperature=0.2,
        grammar=get_text_only_grammar(),
        stop=["<|im_end|>", "<|im_start|>", "[", "{"]
    )
    raw_output = response["choices"][0]["message"]["content"].strip()

    if raw_output.startswith("[") or raw_output.startswith("{"):
        return "I found relevant information but had trouble formatting it. Could you rephrase your question?"

    return raw_output


def chat_completion(messages: List[Dict[str, str]], farm_id: str = "default_farm", thread_id: str = None) -> str:
    llm = get_llm()
    if llm is None:
        return mock_router(messages)

    if is_simple_greeting(messages):
        return "Hello! How can I assist you today?"

    db_summary = get_system_context_summary(farm_id=farm_id)
    last_user_prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    lang_directive = detect_language_instruction(last_user_prompt)

    system_message = {
        "role": "system",
        "content": (
            "You are FarmHand AI, a strict backend routing agent.\n"
            "OUTPUT: Always output valid JSON in one of these formats:\n\n"
            '1. Tool Call: {"action": "tool_call", "data": [{"function_name": "tool_name", "arguments": {...}}], "confidence": 0.95}\n'
            '2. Direct Answer: {"action": "respond", "text": "Your answer here", "confidence": 0.9}\n'
            '3. Error: {"action": "error", "message": "Error message", "confidence": 1.0}\n\n'
            "RULES:\n"
            "- Always output valid JSON (never plain text)\n"
            "- For knowledge queries use query_knowledge_base\n"
            "- For greetings, use action: respond\n"
            "- Include confidence score (0-1)\n\n"
            "TOOLS:\n"
            "- list_animals()\n"
            "- get_animal_record(id: str)\n"
            "- get_sensor_data(node_id: str, sensor_type: str)\n"
            "- write_expenditure(category: str, amount: float, description: str)\n"
            "- write_health_log(animal_id: str, event_type: str, notes: str)\n"
            "- query_knowledge_base(search_query: str)\n\n"
            "EXAMPLES:\n"
            'Input: Why are my chickens losing feathers?\n'
            'Output: {"action": "tool_call", "data": [{"function_name": "query_knowledge_base", "arguments": {"search_query": "feather loss chickens"}}], "confidence": 0.95}\n\n'
            'Input: Hello!\n'
            'Output: {"action": "respond", "text": "Hello! How can I assist you today?", "confidence": 0.99}\n\n'
            'Input: List my animals\n'
            'Output: {"action": "tool_call", "data": [{"function_name": "list_animals", "arguments": {}}], "confidence": 0.98}'
        )
    }

    full_messages = [system_message] + messages

    # PASS 1: Native Chat Completion API with unified grammar
    response_pass1 = llm.create_chat_completion(
        messages=full_messages,
        max_tokens=512,
        temperature=0.0,
        grammar=get_unified_grammar(),
        stop=["<|im_end|>", "<|im_start|>"]
    )
    text_pass1 = response_pass1["choices"][0]["message"]["content"].strip()
    print("[llm_engine] Pass 1 Raw Output:", text_pass1)

    is_tool_call, parsed_result = parse_tool_calls(text_pass1)
    final_text = text_pass1

    # Handle direct responses immediately
    if not is_tool_call and parsed_result:
        if parsed_result[0].get("_direct"):
            return parsed_result[0].get("text", "")
        if parsed_result[0].get("_error"):
            return f"Error: {parsed_result[0].get('message', 'Unknown')}"

    # PASS 2: Grammar Catch (fallback)
    is_greeting = final_text.strip().lower().startswith("hello") or is_simple_greeting(messages)
    if not is_tool_call and not is_greeting:
        print("[llm_engine] Pass 1 failed. Enforcing Grammar Pass 2...")
        grammar = get_unified_grammar()
        response_pass2 = llm.create_chat_completion(
            messages=full_messages,
            max_tokens=512,
            temperature=0.0,
            grammar=grammar,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        text_pass2 = response_pass2["choices"][0]["message"]["content"].strip()
        print("[llm_engine] Pass 2 Grammar Output:", text_pass2)
        is_tool_call, parsed_result = parse_tool_calls(text_pass2)

        # Handle direct responses in Pass 2
        if not is_tool_call and parsed_result:
            if parsed_result[0].get("_direct"):
                return parsed_result[0].get("text", "")
            if parsed_result[0].get("_error"):
                return f"Error: {parsed_result[0].get('message', 'Unknown')}"

        final_text = text_pass2

    # Extract tool calls from parsed result
    tool_calls = [t for t in parsed_result if isinstance(t, dict) and "function_name" in t]

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

        # PASS 3: Totally Stateless Extraction
        if rag_context_prompt:
            if not retrieved_chunks:
                return "I couldn't find any relevant information in the knowledge base to answer your question."
            
            rag_text = generate_stateless_answer(llm, rag_context_prompt, last_user_prompt, lang_directive)
            print(f"[llm_engine] RAG stateless output: {rag_text}")
            return sanitize_response(rag_text)

        # PASS 3: Database Stateless Extraction
        tool_feedback_str = "\n".join([f"- {tr['tool']}: {json.dumps(tr['result'])}" for tr in tool_results])
        synth_text = generate_stateless_answer(llm, tool_feedback_str, last_user_prompt, lang_directive)
        
        return sanitize_response(synth_text)

    if not is_tool_call and not is_greeting:
        return "I encountered an error mapping that request. Could you rephrase it?"

    return sanitize_response(final_text)