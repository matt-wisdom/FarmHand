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

COMPRESSION_THRESHOLD = 3000
COMPRESSION_TARGET = 1000

_thread_token_counts: Dict[str, int] = {}


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def extractive_summary(messages: List[Dict], max_tokens: int = COMPRESSION_TARGET) -> str:
    """Extract key sentences using TF-IDF-like scoring."""
    all_text = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        all_text += f"{role}: {content}\n"

    sentences = all_text.replace("\n", ". ").split(". ")
    if len(sentences) <= 4:
        return all_text

    scored = []
    for sent in sentences:
        if len(sent) < 10:
            continue
        score = len(sent) // 10
        for other in sentences:
            if sent != other and sent in other:
                score -= 1
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored[:6]]
    selected.sort(key=lambda s: all_text.find(s))

    summary = ". ".join(selected)
    if len(summary) > max_tokens * 4:
        summary = summary[:max_tokens * 4]
    return summary


def compress_thread_messages(messages: List[Dict], thread_id: str) -> List[Dict]:
    """Compress conversation history, prepend summary to system prompt."""
    if not messages:
        return messages

    has_system = messages[0].get("role") == "system"
    if has_system and len(messages) <= 4:
        return messages

    summary = extractive_summary(messages[1:] if has_system else messages)

    new_messages = []
    if has_system:
        original_system = messages[0]
        new_system = {
            "role": "system",
            "content": original_system["content"]
            + f"\n\n[Prior conversation summary: {summary}]"
        }
        new_messages.append(new_system)
    else:
        new_messages.append({
            "role": "system",
            "content": f"[Prior conversation summary: {summary}]"
        })

    last_4 = messages[-4:] if len(messages) > 4 else messages
    for m in last_4:
        if has_system and m == messages[0]:
            continue
        new_messages.append(m)

    print(f"[llm_engine] Compressed thread {thread_id[:8]}: {len(messages)} → {len(new_messages)} messages")
    return new_messages


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


def build_response_json_schema() -> Dict[str, Any]:
    """Schema for constrained JSON responses from Pass 1."""
    tool_names = list(TOOL_MAP.keys())
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tool_call", "respond", "error"]
            },
            "data": {
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
            "text": {"type": "string"},
            "message": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["action"],
        "additionalProperties": False
    }


def get_llama_grammar() -> LlamaGrammar:
    global _llama_grammar_instance
    if _llama_grammar_instance is None:
        schema_str = json.dumps(build_tools_json_schema())
        _llama_grammar_instance = LlamaGrammar.from_json_schema(schema_str)
    return _llama_grammar_instance


_response_grammar_instance = None


def get_response_grammar() -> LlamaGrammar:
    """Grammar for constrained JSON responses (action/data/text format)."""
    global _response_grammar_instance
    if _response_grammar_instance is None:
        schema_str = json.dumps(build_response_json_schema())
        _response_grammar_instance = LlamaGrammar.from_json_schema(schema_str)
    return _response_grammar_instance


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
    """Parse both legacy and new JSON response formats."""
    clean_text = output_text.strip()

    # Try new format: {"action": "tool_call", "data": [...]}
    if clean_text.startswith("{"):
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict):
                action = data.get("action")
                if action == "tool_call" and "data" in data:
                    tool_calls = []
                    for item in data.get("data", []):
                        if isinstance(item, dict) and "function_name" in item:
                            fn_name = item["function_name"]
                            if fn_name in TOOL_MAP:
                                args = item.get("arguments", {})
                                tool_calls.append({"function_name": fn_name, "arguments": args})
                    if tool_calls:
                        return True, tool_calls
                elif action == "respond":
                    # Direct response - return special marker
                    return False, [{"_direct_response": data.get("text", ""), "_confidence": data.get("confidence", 0.9)}]
                elif action == "error":
                    return False, [{"_error": data.get("message", "Unknown error")}]
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
    """Two-pass: Extract facts first, then format into answer."""

    # Pass 3a: Extract key facts
    extract_prompt = {
        "role": "system",
        "content": (
            "You are a factual extractor. Read the reference material and extract 2-3 key facts relevant to the question.\n"
            f"{lang_directive}\n"
            "Output only simple facts as bullet points. No JSON. No tool calls."
        )
    }

    extract_user = {
        "role": "user",
        "content": f"Reference:\n{context_data}\n\nQuestion: {user_question}\n\nKey facts:"
    }

    response1 = llm.create_chat_completion(
        messages=[extract_prompt, extract_user],
        max_tokens=256,
        temperature=0.1,
        stop=["<|im_end|>", "<|im_start|>", "[", "{"]
    )
    facts = response1["choices"][0]["message"]["content"].strip()
    print(f"[llm_engine] Pass 3a facts: {facts[:100]}")

    # If extraction returned JSON, use raw context
    if facts.startswith("[") or facts.startswith("{") or not facts:
        facts = context_data[:500]

    # Pass 3b: Format into final answer
    format_prompt = {
        "role": "system",
        "content": (
            "You are FarmHand AI, a helpful farming assistant.\n"
            f"{lang_directive}\n"
            "Give a direct, friendly answer in 1-3 sentences. If you don't know, say so."
        )
    }

    format_user = {
        "role": "user",
        "content": f"Facts: {facts}\n\nQuestion: {user_question}\n\nAnswer:"
    }

    response2 = llm.create_chat_completion(
        messages=[format_prompt, format_user],
        max_tokens=256,
        temperature=0.2,
        stop=["<|im_end|>", "<|im_start|>", "[", "{"]
    )
    answer = response2["choices"][0]["message"]["content"].strip()
    print(f"[llm_engine] Pass 3b answer: {answer[:100]}")

    return answer if answer else "I found relevant information but couldn't format the response."


def chat_completion(messages: List[Dict[str, str]], farm_id: str = "default_farm", thread_id: str = None) -> str:
    global _thread_token_counts

    if thread_id:
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if total_tokens > COMPRESSION_THRESHOLD:
            messages = compress_thread_messages(messages, thread_id)
            new_total = sum(estimate_tokens(m.get("content", "")) for m in messages)
            _thread_token_counts[thread_id] = new_total
            print(f"[llm_engine] Compressed thread {thread_id[:8]}: {total_tokens} → {new_total} tokens")
        else:
            _thread_token_counts[thread_id] = total_tokens

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
            "OUTPUT FORMAT: You MUST output valid JSON in one of these formats:\n\n"
            '1. Tool Call: {"action": "tool_call", "data": [{"function_name": "tool_name", "arguments": {...}}], "confidence": 0.95}\n'
            '2. Direct Response: {"action": "respond", "text": "Your answer here", "confidence": 0.9}\n'
            '3. Error: {"action": "error", "message": "Error message", "confidence": 1.0}\n\n'
            "RULES:\n"
            "- Always output valid JSON, never plain text\n"
            "- For knowledge queries, use query_knowledge_base tool\n"
            "- For greetings, use action: respond with text: greeting\n"
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
            'Input: List all my animals\n'
            'Output: {"action": "tool_call", "data": [{"function_name": "list_animals", "arguments": {}}], "confidence": 0.98}'
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

    is_tool_call, parsed_result = parse_tool_calls(text_pass1)
    final_text = text_pass1

    # Check for direct responses from new format
    if not is_tool_call and parsed_result:
        if "_direct_response" in parsed_result[0]:
            return parsed_result[0]["_direct_response"]
        if "_error" in parsed_result[0]:
            return f"Error: {parsed_result[0]['_error']}"

    # PASS 2: Aggressive Grammar Catch
    is_greeting = final_text.strip().lower().startswith("hello") or is_simple_greeting(messages)
    if not is_tool_call and not is_greeting:
        print("[llm_engine] Pass 1 failed to route. Enforcing Grammar Pass 2...")
        grammar = get_response_grammar()
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

        # Check for direct responses in Pass 2
        if not is_tool_call and parsed_result:
            if "_direct_response" in parsed_result[0]:
                return parsed_result[0]["_direct_response"]
            if "_error" in parsed_result[0]:
                return f"Error: {parsed_result[0]['_error']}"

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

# PASS 3: RAG Answer Generation (two-pass: extract facts, then format)
        if rag_context_prompt:
            if not retrieved_chunks:
                return "I couldn't find any relevant information in the knowledge base to answer your question."

            # Clean context to reduce confusion
            clean_context = rag_context_prompt.replace("[{", "START ").replace("}]", " END")

            rag_text = generate_stateless_answer(llm, clean_context, last_user_prompt, lang_directive)

            if not rag_text or len(rag_text.strip()) < 5:
                return "Error processing knowledge base response. Please rephrase your question."

            return sanitize_response(rag_text)

        # PASS 3: Database Stateless Extraction
        tool_feedback_str = "\n".join([f"- {tr['tool']}: {json.dumps(tr['result'])}" for tr in tool_results])
        synth_text = generate_stateless_answer(llm, tool_feedback_str, last_user_prompt, lang_directive)
        
        return sanitize_response(synth_text)

    # Fallback for non-tool, non-greeting inputs that couldn't be parsed
    return "I couldn't understand that request. Try asking about your farm animals, expenditures, health logs, or general farming questions."