import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

from database import (
    get_current_flock_totals,
    get_farm_by_id,
    get_flock_count_on_date,
    get_system_context_summary,
    normalize_species_name,
    record_flock_event,
)
from tool_registry import TOOL_MAP, TOOL_SCHEMAS, execute_tool
from rag_pipeline import search_knowledge_base
from translator import translate_en_to_ha, translate_ha_to_en

logger = logging.getLogger("FarmHandEngine")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "qwen2.5-3b-instruct.Q4_K_M.gguf"

# 2 physical CPU cores on Intel Core i7-7500U to maximize CPU throughput
N_CTX = 4096
N_THREADS = 2

# --- FIX A: minimum similarity score required to trust a RAG hit ---
# Tune this against your embedding model's actual score distribution. If your
# search_knowledge_base returns cosine similarity in [0,1], 0.55 is a reasonable
# starting floor; if it returns raw distances (lower = better), invert this logic.
RAG_MIN_SCORE = 0.55

_llm_instance: Optional[Llama] = None
_llama_grammar_instance: Optional[LlamaGrammar] = None
_anti_json_logit_bias: Optional[Dict[int, float]] = None
_english_logit_bias: Optional[Dict[int, float]] = None


def normalize_language(lang: Optional[str]) -> str:
    """Normalize language code/name to 'english', 'hausa', or 'pidgin'."""
    if not lang:
        return "english"
    l = str(lang).strip().lower()
    if l in ["ha", "hausa"] or l.startswith("ha-") or "hausa" in l:
        return "hausa"
    if l in ["pg", "pidgin", "pcm"] or l.startswith("pid") or "pidgin" in l:
        return "pidgin"
    return "english"


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


def get_llama_grammar() -> Optional[LlamaGrammar]:
    global _llama_grammar_instance
    if _llama_grammar_instance is None:
        schema = build_tools_json_schema()
        schema_json = json.dumps(schema)
        _llama_grammar_instance = LlamaGrammar.from_json_schema(schema_json)
    return _llama_grammar_instance


def get_llm() -> Optional[Llama]:
    """Lazy loader for llama.cpp model instance."""
    global _llm_instance, _anti_json_logit_bias, _english_logit_bias
    if _llm_instance is None:
        if MODEL_PATH.exists():
            print(f"[llm_engine] Loading llama.cpp model from {MODEL_PATH} (threads={N_THREADS}, ctx={N_CTX})...")
            _llm_instance = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                chat_format="chatml",
                verbose=False
            )
            # Token bias against '[' and '{' to prevent default tool-calling syntax in Pass 3
            sq_tokens = _llm_instance.tokenize(b'[')
            cu_tokens = _llm_instance.tokenize(b'{')
            _anti_json_logit_bias = {tok: -100.0 for tok in sq_tokens + cu_tokens}

            # English-mode bias: suppress JSON tokens + Pidgin particles directly in llama.cpp
            pidgin_words = [
                'dey', ' dey', 'Dey', ' Dey', 'wey', ' wey', 'Wey', 'well-well', ' well-well',
                'wetin', ' wetin', 'Wetin', 'una', ' una', 'dem', ' dem', 'sabi', ' sabi'
            ]
            _english_logit_bias = dict(_anti_json_logit_bias)
            for w in pidgin_words:
                toks = _llm_instance.tokenize(w.encode('utf-8'), add_bos=False)
                for t in toks:
                    _english_logit_bias[t] = -100.0
        else:
            print(f"[llm_engine] Model file not found at {MODEL_PATH}")
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


def is_conversational_greeting(query: str) -> bool:
    """Robust detection strictly for conversational greetings, chit-chat, and polite expressions."""
    q = query.lower().strip()
    words = set(re.findall(r'\b\w+\b', q))
    if not words:
        return True

    greeting_words = {
        'hello', 'hi', 'hey', 'sannu', 'barka', 'kwana', 'morning', 'afternoon',
        'evening', 'yaya', 'greetings', 'greeting', 'welcome', 'thanks', 'thank',
        'nagode', 'gode', 'howfar', 'sup', 'yo'
    }
    agri_action_words = {
        'chicken', 'chickens', 'goat', 'goats', 'cattle', 'cow', 'cows', 'sheep', 'ram', 'pig', 'pigs',
        'kaji', 'kaza', 'awaki', 'akuya', 'shanu', 'saniya', 'tumaki',
        'cough', 'coughing', 'sneeze', 'sneezing', 'die', 'dying', 'dead', 'sick', 'sickness', 'disease',
        'fever', 'limp', 'limping', 'cuta', 'ciwo', 'tari', 'mura', 'zazzabi', 'magani',
        'blister', 'blisters', 'scab', 'scabs', 'wound', 'wounds', 'pox', 'fungal', 'fungus',
        'feed', 'housing', 'pen', 'coop', 'vaccine', 'vaccination', 'egg', 'eggs', 'weight',
        'cost', 'expense', 'spent', 'buy', 'bought', 'naira', 'kudi', 'count', 'number', 'many',
        'animal', 'animals', 'flock', 'herd', 'livestock'
    }

    has_greeting = bool(words.intersection(greeting_words)) or 'how far' in q or 'how are you' in q or 'how is work' in q or 'ina kwana' in q
    has_action = bool(words.intersection(agri_action_words))

    return has_greeting and not has_action


def handle_fast_intent(query: str, language: str) -> Optional[str]:
    """Strictly handles conversational greetings in 0.00s."""
    q = query.lower().strip()

    if not is_conversational_greeting(q):
        return None

    if "gode" in q or "thank" in q:
        if language == "hausa":
            return "Barka da aiki! Idan kuna da wata tambaya game da dabbobinku ko ayyukan gonarku, ina nan a shirye in taimaka muku."
        elif language == "pidgin":
            return "You are welcome! If you get any other question about your farm or animals, just ask me anytime."
        return "You are welcome! If you have any other questions regarding your livestock, poultry, or farm operations, feel free to ask anytime."

    if language == "hausa":
        return "Sannu! Barka da asuba. Ni ne FarmHand AI, mataimakin gonarka. Yaya zan iya taimaka maka da dabbobinka ko ayyukan gonarka a yau?"
    elif language == "pidgin":
        return "Hello! I be FarmHand AI. How I fit help you with your animals or farm today?"
    return "Hello! I am FarmHand AI, your digital agricultural assistant. How can I assist you with your livestock, poultry, or farm operations today?"


def format_database_tool_context(tool_results: List[Dict[str, Any]]) -> str:
    lines = []
    for tr in tool_results:
        tool = tr["tool"]
        res = tr["result"]

        if tool == "list_animals" and isinstance(res, dict):
            data = res.get("data", [])
            total = res.get("total", sum(a.get("count", 0) for a in data) if isinstance(data, list) else 0)
            lines.append("FARM FLOCK INVENTORY QUERY RESULT:")
            if isinstance(data, dict):
                # Historical date query
                lines.append(f"- Historical Count on {data.get('as_of_date')}: {data.get('count', data.get('total_flock_size', 0))} {data.get('species', 'animals')}")
            elif data:
                flock_strs = [f"{a.get('species', 'Unknown').capitalize()}: {a.get('count', 0)}" for a in data]
                lines.append(f"- Current Flock Counts: {', '.join(flock_strs)} (Total: {total})")
            else:
                lines.append("- Current Inventory: None (0 animals recorded in database)")

        elif tool == "register_flock" and isinstance(res, dict):
            lines.append("FLOCK REGISTRATION RESULT:")
            if res.get("status") == "success":
                entry = res.get("entry", {})
                lines.append(f"- Successfully recorded {entry.get('event_type')} of {entry.get('count_change', 0):+d} {entry.get('species')}. New total flock balance is {entry.get('new_total', 0)}.")
            else:
                lines.append(f"- Failed to record flock: {res.get('message', 'Unknown error')}")
                lines.append(f"- Failed to record flock: {res.get('message', 'Unknown error')}")

        elif tool == "list_expenditures" and isinstance(res, dict):
            cnt = res.get("count", 0)
            data = res.get("data", [])
            lines.append("FARM EXPENDITURES QUERY RESULT:")
            lines.append(f"- Total Recorded Expenditures: {cnt}")
            if data:
                exp_strs = [f"ID {e.get('id')}: NGN {e.get('amount', 0):,.2f} for {e.get('category')} - {e.get('description')}" for e in data[:5]]
                lines.append(f"- Recent Expenses: {'; '.join(exp_strs)}")
            else:
                lines.append("- Recent Expenses: None recorded")

        elif tool == "list_health_logs" and isinstance(res, dict):
            cnt = res.get("count", 0)
            data = res.get("data", [])
            lines.append("FLOCK HEALTH LOGS QUERY RESULT:")
            lines.append(f"- Total Health Logs: {cnt}")
            if data:
                log_strs = [f"Flock ({h.get('species')}): {h.get('event_type')} - {h.get('notes')}" for h in data[:5]]
                lines.append(f"- Health History: {'; '.join(log_strs)}")
            else:
                lines.append("- Health History: None recorded")

        elif tool == "log_farm_observation" and isinstance(res, dict):
            mem = res.get("memory", {})
            lines.append("FARM CLINICAL OBSERVATION LOGGED:")
            lines.append(f"- Saved observation for {mem.get('species', 'Flock')} ({mem.get('category', 'symptom')}): {mem.get('observation')}")

        else:
            lines.append(f"- {tool}: {json.dumps(res)}")

    return "\n".join(lines)


# --- FIX A helper: filter RAG hits by a relevance floor before trusting them ---
def filter_relevant_rag_hits(rag_hits: List[Dict[str, Any]], min_score: float = RAG_MIN_SCORE) -> List[Dict[str, Any]]:
    """
    Drops RAG hits that don't clear a minimum relevance score, so an off-topic
    query (e.g. an inventory/database question) doesn't get "answered" using
    unrelated husbandry/disease chunks the retriever returned anyway.

    Assumes higher score = more relevant (cosine similarity style). If your
    search_knowledge_base returns a distance metric instead (lower = better),
    flip the comparison below.
    """
    if not rag_hits:
        return []

    filtered = []
    for h in rag_hits:
        score = h.get("score")
        # If the pipeline doesn't report a score at all, we can't apply the floor
        # reliably — keep the hit but log it so this gets noticed and fixed upstream.
        if score is None:
            print(f"[llm_engine] WARNING: RAG hit for '{h.get('filename')}' has no 'score' field; "
                  f"cannot apply relevance floor. Consider updating search_knowledge_base to return scores.")
            filtered.append(h)
            continue
        if score >= min_score:
            filtered.append(h)
        else:
            print(f"[llm_engine] Dropping low-relevance RAG hit '{h.get('filename')}' (score={score:.3f} < {min_score})")

    return filtered


def generate_stateless_answer(llm: Llama, context_data: str, user_question: str, norm_lang: str, db_summary: str) -> str:
    """Pass 3: Natural language synthesis using strictly positive prompting and API Chat Wrapper."""

    # Budget context length to prevent exceeding token limits
    safe_context = context_data[:2500].strip() if context_data else ""

    if norm_lang == "pidgin":
        system_prompt = (
            "You are FarmHand AI, an expert agricultural and veterinary advisor.\n"
            "Respond to the farmer ONLY in warm, natural Nigerian Pidgin English.\n\n"
            f"FARM INVENTORY & PROFILE:\n{db_summary}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer based strictly on the Reference Context.\n"
            "- State the exact facts, advice, or steps directly."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Reference Context:\n{safe_context}\n\nFarmer Question:\n{user_question}"}
        ]
    else:
        system_prompt = (
            "You are FarmHand AI, an expert agricultural and veterinary specialist.\n"
            "Respond to the farmer in clear, professional, standard international English.\n\n"
            f"FARM INVENTORY & PROFILE:\n{db_summary}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer the farmer's question directly, clearly, and concisely based ONLY on facts present in the Reference Context.\n"
            "- State specific clinical treatments, medications, active ingredients, dosage guidance, and management steps mentioned.\n"
            "- Do NOT use Nigerian Pidgin, slang, or colloquial expressions.\n"
            "- Do NOT ask conversational follow-up questions or delay your answer."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Reference Context:\n- Treat tick infestations with approved acaricides (e.g. Amitraz spray or Cypermethrin dip).\n- Keep goat pens dry and well-ventilated.\n\nFarmer Question:\nHow do I manage ticks on my goats?"
            },
            {
                "role": "assistant",
                "content": "To manage ticks on goats, apply an approved acaricide spray or dipping solution such as Amitraz or Cypermethrin according to the manufacturer dosage. Inspect animals regularly and keep housing clean and dry to reduce tick exposure."
            },
            {"role": "user", "content": f"Reference Context:\n{safe_context}\n\nFarmer Question:\n{user_question}"}
        ]

    # Calculate token headroom dynamically
    est_prompt_tokens = len(system_prompt + safe_context + user_question) // 3
    max_gen_tokens = max(64, min(300, N_CTX - est_prompt_tokens - 100))

    bias = _english_logit_bias if norm_lang != "pidgin" else _anti_json_logit_bias

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_gen_tokens,
        temperature=0.1,
        repeat_penalty=1.15,
        logit_bias=bias,
        stop=["<|im_end|>", "<|im_start|>"]
    )

    return response["choices"][0]["message"]["content"].strip()


def format_tool_direct_response(tool_name: str, result: dict, farm_id: str, language: str = "english") -> Optional[str]:
    """Generates an immediate, 100% accurate grounded response for database operations."""
    farm = get_farm_by_id(farm_id)
    farm_name = farm["name"] if farm else "your farm"

    if tool_name == "list_animals":
        if result.get("historical"):
            d = result.get("data", {})
            date_str = result.get("date", "specified date")
            cnt = d.get("count", d.get("total_flock_size", 0))
            sp = d.get("species", "animals")
            if language == "hausa":
                return f"Bisa ga bayanan gonarku ({farm_name}) a ranar {date_str}: kuna da {sp} {cnt}."
            elif language == "pidgin":
                return f"According to your records ({farm_name}) on {date_str}: you get {cnt} {sp}."
            return f"According to your flock ledger for {farm_name} as of {date_str}: you had {cnt} {sp}."
        else:
            data = result.get("data", [])
            total = result.get("total", 0)
            if not data or total == 0:
                desc = farm.get("description", "") if farm else ""
                notes = f' (Farm profile notes: "{desc}")' if desc else ""
                if language == "hausa":
                    return f"Bisa ga bayanan gonarku ({farm_name}), a halin yanzu kuna da dabbobi 0 da aka rubuta a rumbun bayanan.{notes}"
                elif language == "pidgin":
                    return f"According to your farm records ({farm_name}), you get 0 animals recorded right now.{notes}"
                return f"According to your flock ledger for {farm_name}, you currently have 0 registered animals recorded.{notes}"
            summary = ", ".join([f"{a.get('species', 'Unknown').capitalize()}: {a.get('count', 0)}" for a in data])
            if language == "hausa":
                return f"Bisa ga bayanan gonarku ({farm_name}), a halin yanzu kuna da jimillar dabbobi {total}: {summary}."
            elif language == "pidgin":
                return f"According to your farm records ({farm_name}), you get {total} animals right now: {summary}."
            return f"According to your flock ledger for {farm_name}, you currently have {total} animals: {summary}."

    elif tool_name == "register_flock":
        entry = result.get("entry", {})
        sp = entry.get("species", "animals")
        evt = entry.get("event_type", "update")
        chg = entry.get("count_change", 0)
        tot = entry.get("new_total", 0)
        if language == "hausa":
            return f"An rubuta: An sabunta {sp} ({evt}) a bayanan gonarku ({farm_name}). Yawan su a yanzu: {tot}."
        elif language == "pidgin":
            return f"Recorded: Updated {sp} ({evt}) for your farm ({farm_name}). Your new total na {tot}."
        return f"Recorded: Logged {evt} ({chg:+d} {sp}). Your current flock total for {farm_name} is now {tot}."

    elif tool_name == "list_expenditures":
        cnt = result.get("count", 0)
        data = result.get("data", [])
        total_amt = sum(e.get("amount", 0) for e in data)
        if language == "hausa":
            return f"Bisa ga bayanan kudaden gonarku ({farm_name}), kuna da rubuce-rubucen kudaden da aka kashe guda {cnt} da suka kai NGN {total_amt:,.2f}."
        elif language == "pidgin":
            return f"According to your farm expenses ({farm_name}), you get {cnt} records wey reach NGN {total_amt:,.2f}."
        return f"According to your financial records for {farm_name}, you have {cnt} recorded expenditures totaling NGN {total_amt:,.2f}."

    elif tool_name == "write_expenditure":
        data = result.get("data", {})
        amt = data.get("amount", 0)
        cat = data.get("category", "operations")
        desc = data.get("description", "")
        return f"Recorded: Expenditure of NGN {amt:,.2f} logged under '{cat}' for {farm_name} ({desc})."

    elif tool_name == "list_health_logs":
        cnt = result.get("count", 0)
        data = result.get("data", [])
        return f"Found {cnt} health log records in your database for {farm_name}."

    elif tool_name == "write_health_log":
        data = result.get("data", {})
        return f"Recorded: Health event '{data.get('event_type')}' logged successfully for animal {data.get('animal_id')}."

    elif tool_name == "log_farm_observation":
        mem = result.get("memory", {})
        sp = mem.get("species", "Livestock")
        obs = mem.get("observation", "")
        if language == "hausa":
            return f"An rubuta lura a bayanan gonar {farm_name}: {sp} - {obs}."
        elif language == "pidgin":
            return f"I don log this observation for your farm {farm_name} memory: {sp} - {obs}."
        return f"Recorded into persistent memory for {farm_name}: {sp} - {obs}."

    return None


def chat_completion(
    messages: List[Dict[str, str]],
    farm_id: str = "default_farm",
    thread_id: Optional[str] = None,
    language: str = "english"
) -> str:
    """Unified multi-turn chat completion for FarmHand AI."""
    turn_start = time.time()
    norm_lang = normalize_language(language)
    print(f"\n[llm_engine] --- Chat Turn Start | Farm: '{farm_id}' | Lang: '{norm_lang}' ---")

    user_prompts = [m.get("content", "").strip() for m in messages if m.get("role") == "user"]
    last_user_query = user_prompts[-1] if user_prompts else ""

    # Step 1: Translate Hausa input to English for knowledge retrieval if needed
    effective_messages = []
    if norm_lang == "hausa":
        print(f"[llm_engine] Translating user input from Hausa to English...")
        for m in messages:
            if m.get("role") == "user":
                effective_messages.append({"role": "user", "content": translate_ha_to_en(m.get("content", ""))})
            else:
                effective_messages.append(m)
        current_query_en = effective_messages[-1]["content"] if effective_messages else last_user_query
    else:
        effective_messages = list(messages)
        current_query_en = last_user_query

    # Step 2: Instant Grounded Fast Intent Handler strictly for greetings
    fast_ans = handle_fast_intent(current_query_en, language=norm_lang)
    if fast_ans:
        print(f"[llm_engine] Fast greeting handler answered in {time.time() - turn_start:.2f}s")
        return fast_ans

    llm = get_llm()
    if llm is None:
        return "[Fallback] Model not loaded."

    farm_summary = get_system_context_summary(farm_id)

    # Step 3: PASS 1 - JSON ROUTING (No logit bias applied here so it CAN output JSON)
    routing_system = {
        "role": "system",
        "content": (
            "You are the tool routing engine for FarmHand AI.\n"
            "Output ONLY a valid JSON array with the single best tool call.\n\n"
            "TOOLS:\n"
            "- list_animals(species: str, date_str: str): Check or ask for animal headcount, how many animals/birds/goats/cows/sheep are on the farm, or count on a past date.\n"
            "- register_flock(species: str, count: int, event_type: str, notes: str): Set, record, or update animal headcount (e.g. 'I have 5 chickens', 'We currently have 9 goats', 'bought 10 cows', '2 birds died').\n"
            "- list_expenditures(category: str): View recorded farm expenses or spending.\n"
            "- write_expenditure(category: str, amount: float, description: str): Record a new financial expense.\n"
            "- log_farm_observation(species: str, observation: str, category: str): Record an observed physical symptom, abnormal movement, disease sign, or medication into farm memory.\n"
            "- query_knowledge_base(search_query: str): Ask about diseases, illness, symptoms, treatments, medication, vaccines, feeding, or farming advice.\n\n"
            "SPECIES MAPPING:\n"
            "- chickens / hens / broilers / birds -> \"poultry\"\n"
            "- goats / kids / bucks -> \"goat\"\n"
            "- cows / cattle / bulls / calves -> \"cattle\"\n"
            "- sheep / rams / ewes / lambs -> \"sheep\"\n"
            "- pigs / swine / piglets -> \"pig\"\n"
            "- fish / catfish / tilapia -> \"fish\"\n\n"
            "EXAMPLES:\n"
            "Farmer: 'how many chickens do i have' -> [{\"function_name\": \"list_animals\", \"arguments\": {\"species\": \"poultry\"}}]\n"
            "Farmer: 'how many goats do i have' -> [{\"function_name\": \"list_animals\", \"arguments\": {\"species\": \"goat\"}}]\n"
            "Farmer: 'how many animals do i have' -> [{\"function_name\": \"list_animals\", \"arguments\": {}}]\n"
            "Farmer: 'We currently have 9 goats' -> [{\"function_name\": \"register_flock\", \"arguments\": {\"species\": \"goat\", \"count\": 9, \"event_type\": \"initial_count\", \"notes\": \"\"}}]\n"
            "Farmer: 'I have 5 chickens' -> [{\"function_name\": \"register_flock\", \"arguments\": {\"species\": \"poultry\", \"count\": 5, \"event_type\": \"initial_count\", \"notes\": \"\"}}]\n"
            "Farmer: 'I bought 10 cows' -> [{\"function_name\": \"register_flock\", \"arguments\": {\"species\": \"cattle\", \"count\": 10, \"event_type\": \"purchase\", \"notes\": \"\"}}]\n"
            "Farmer: 'We bought 15 sheep' -> [{\"function_name\": \"register_flock\", \"arguments\": {\"species\": \"sheep\", \"count\": 15, \"event_type\": \"purchase\", \"notes\": \"\"}}]\n"
            "Farmer: '3 chickens died today' -> [{\"function_name\": \"register_flock\", \"arguments\": {\"species\": \"poultry\", \"count\": -3, \"event_type\": \"mortality\", \"notes\": \"\"}}]\n"
            "Farmer: 'my goat is moving weird' -> [{\"function_name\": \"query_knowledge_base\", \"arguments\": {\"search_query\": \"goat abnormal movement causes\"}}]\n"
            "Farmer: 'what causes coughing in goats' -> [{\"function_name\": \"query_knowledge_base\", \"arguments\": {\"search_query\": \"goat coughing causes treatment\"}}]\n"
            "Farmer: 'What should i first do when i get a new chicken?' -> [{\"function_name\": \"query_knowledge_base\", \"arguments\": {\"search_query\": \"new chicken arrival care guidelines\"}}]\n"
            "Farmer: 'how much have i spent this month' -> [{\"function_name\": \"list_expenditures\", \"arguments\": {}}]"
        )
    }

    # Build dialogue context from recent messages if multi-turn
    if len(messages) > 1:
        history_turns = []
        for m in messages[-4:-1]:
            r = "Farmer" if m.get("role") == "user" else "Assistant"
            c = (m.get("content") or "").strip()
            if c:
                history_turns.append(f"{r}: {c}")
        if history_turns:
            dialogue_ctx = "\n".join(history_turns)
            router_query_content = f"Recent Dialogue Context:\n{dialogue_ctx}\n\nCurrent Farmer Query:\n{current_query_en}"
        else:
            router_query_content = current_query_en
    else:
        router_query_content = current_query_en

    routing_messages = [routing_system, {"role": "user", "content": router_query_content}]
    grammar = get_llama_grammar()

    print(f"[llm_engine] Running Pass 1 (Router)...")
    response_pass1 = llm.create_chat_completion(
        messages=routing_messages,
        max_tokens=128,
        temperature=0.0,
        grammar=grammar,
        stop=["<|im_end|>", "<|im_start|>"]
    )

    text_pass1 = response_pass1["choices"][0]["message"]["content"].strip()
    is_tool_call, tool_calls = parse_tool_calls(text_pass1)

    # Step 4: Execute Tools
    tool_results = []
    rag_context = ""
    logged_mem_in_tools = False

    if is_tool_call:
        for call in tool_calls:
            fn_name = call["function_name"]
            fn_args = call.get("arguments", {})
            print(f"[llm_engine] Executing tool '{fn_name}' with args {fn_args}")
            res = execute_tool(fn_name, fn_args, farm_id=farm_id)
            tool_results.append({"tool": fn_name, "result": res})

            if fn_name == "log_farm_observation":
                logged_mem_in_tools = True

            if fn_name == "query_knowledge_base" and isinstance(res, dict) and "context_prompt" in res:
                rag_context = res["context_prompt"]

    # Auto-log clinical observations if user reported physical/behavioral symptoms and not logged yet
    symptom_triggers = ["moving weird", "walk funny", "limp", "cough", "sneeze", "not eating", "refus", "swoll", "diarrhea", "bloody", "blister", "discharge", "lesion", "fever", "pale eye", "paraly", "trembl", "tick", "flea", "worm"]
    q_lower = current_query_en.lower()
    if not logged_mem_in_tools and any(st in q_lower for st in symptom_triggers):
        matched_sp = "General"
        for sp_k, sp_v in [("goat", "Goat"), ("poultry", "Poultry"), ("chicken", "Poultry"), ("bird", "Poultry"), ("cattle", "Cattle"), ("cow", "Cattle"), ("bull", "Cattle"), ("sheep", "Sheep"), ("ram", "Sheep"), ("pig", "Pig")]:
            if sp_k in q_lower:
                matched_sp = sp_v
                break
        try:
            import farm_memory
            farm_memory.log_and_embed_observation(
                farm_id=farm_id,
                species=matched_sp,
                category="symptom",
                observation=last_user_query,
                source="chat_inferred"
            )
        except Exception as e:
            print(f"[llm_engine] Auto-log observation notice: {e}")

    if not is_tool_call and len(current_query_en) > 5:
        print("[llm_engine] Pass 1 yielded no tools, falling back to RAG safety net...")
        rag_hits = search_knowledge_base(current_query_en, top_k=2)
        relevant_hits = filter_relevant_rag_hits(rag_hits)

        if relevant_hits:
            rag_context = "\n---\n".join(
                [f"[{h.get('filename')}]: {h.get('text')[:250]}" for h in relevant_hits if len(h.get("text", "")) > 50]
            )
        else:
            print("[llm_engine] No RAG hits cleared the relevance floor; will not synthesize from noise.")

    # Inject semantically matching farm memories into RAG context
    try:
        import farm_memory
        matching_mems = farm_memory.search_farm_memories(farm_id=farm_id, query=current_query_en, top_k=3)
        if matching_mems:
            mem_block = farm_memory.format_memories_for_rag(matching_mems)
            rag_context = f"{mem_block}\n\n{rag_context}" if rag_context else mem_block
            print(f"[llm_engine] Injected {len(matching_mems)} semantic farm memories into RAG context.")
    except Exception as e:
        print(f"[llm_engine] Semantic memory retrieval notice: {e}")

    # Step 5: SYNTHESIS / RESULT DISPATCH
    if is_tool_call:
        for call in tool_calls:
            fn_name = call["function_name"]
            tool_match = next((tr["result"] for tr in tool_results if tr["tool"] == fn_name), None)
            if tool_match and fn_name in ("list_animals", "register_flock", "list_expenditures", "write_expenditure", "list_health_logs", "write_health_log", "log_farm_observation") and not rag_context:
                db_direct_ans = format_tool_direct_response(fn_name, tool_match, farm_id=farm_id, language=norm_lang)
                if db_direct_ans:
                    print(f"[llm_engine] Database tool '{fn_name}' direct response generated.")
                    total_time = time.time() - turn_start
                    print(f"[llm_engine] TOTAL TURN TIME: {total_time:.2f}s\n")
                    return db_direct_ans

    if rag_context:
        print(f"[llm_engine] Running Pass 3 (RAG Synthesis)...")
        raw_output = generate_stateless_answer(llm, rag_context, router_query_content, norm_lang, farm_summary)
    elif tool_results:
        print(f"[llm_engine] Running Pass 3 (Database Synthesis)...")
        db_context = format_database_tool_context(tool_results)
        raw_output = generate_stateless_answer(llm, db_context, router_query_content, norm_lang, farm_summary)
    else:
        raw_output = "I couldn't process that command. Could you please rephrase what you need help with?"

    # Step 6: Post-process & Language formatting
    if norm_lang == "hausa":
        print(f"[llm_engine] Translating English response to Hausa...")
        ha_translated = translate_en_to_ha(raw_output)
        religious_artifacts = ["littafi mai tsarki", "ãdalci", "sikẽlin", "la'ĩmi", "al'ummai", "karin magana"]
        if any(art in ha_translated.lower() for art in religious_artifacts):
            print(f"[llm_engine] Detected MarianMT religious artifact in translation. Falling back to English.")
            final_output = raw_output
        else:
            final_output = ha_translated
    else:  # English or Pidgin
        final_output = raw_output

    total_time = time.time() - turn_start
    print(f"[llm_engine] TOTAL TURN TIME: {total_time:.2f}s\n")
    return final_output