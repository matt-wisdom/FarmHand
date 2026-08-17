import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama

from database import get_farm_by_id, get_system_context_summary
from rag_pipeline import search_knowledge_base
from translator import translate_en_to_ha, translate_ha_to_en

logger = logging.getLogger("FarmHandEngine")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "qwen2.5-3b-instruct.Q4_K_M.gguf"

# 2 physical CPU cores on Intel Core i7-7500U to maximize CPU throughput
N_CTX = 2048
N_THREADS = 2

_llm_instance: Optional[Llama] = None


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


def get_llm() -> Optional[Llama]:
    """Lazy loader for llama.cpp model instance."""
    global _llm_instance
    if _llm_instance is None:
        if MODEL_PATH.exists():
            print(f"[llm_engine] Loading llama.cpp model from {MODEL_PATH} (threads={N_THREADS}, ctx={N_CTX})...")
            _llm_instance = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                verbose=False
            )
        else:
            print(f"[llm_engine] Model file not found at {MODEL_PATH}")
            _llm_instance = None
    return _llm_instance


def convert_pidgin_to_clean_english(text: str) -> str:
    """Converts fine-tuned Nigerian Pidgin idioms into standard English for clean translation."""
    replacements = [
        (r'\bi dey help you with\b', 'I can assist you with'),
        (r'\bi dey here to help you\b', 'I am here to assist you'),
        (r'\bi dey well-well\b', 'I am doing well'),
        (r'\bi dey\b', 'I am'),
        (r'\bmake you\b', 'please'),
        (r'\bmake we\b', 'let us'),
        (r'\bwell-well\b', 'properly'),
        (r'\bwell-quick-quick\b', 'quickly'),
        (r'\bno fit\b', 'cannot'),
        (r'\bno dey\b', 'does not'),
        (r'\bdey show say\b', 'shows that'),
        (r'\bdey enter for pen\b', 'in the pen'),
        (r'\bdey recorded for\b', 'recorded in'),
        (r'\bwey dey\b', 'that is'),
        (r'\bna so\b', 'that is correct'),
        (r'\babeg\b', 'please'),
        (r'\bwetin\b', 'what'),
        (r'\byou fit\b', 'you can'),
        (r'\byou get\b', 'you have'),
        (r'\bif you get\b', 'if you have'),
        (r'\bif dem dey\b', 'if they are'),
        (r'\bdem dey\b', 'they are'),
        (r'\bdem be\b', 'it may be'),
        (r'\bdey worse\b', 'get worse'),
        (r'\bquick-quick\b', 'quickly'),
        (r'\be dey help\b', 'it helps'),
        (r'\biver go use\b', 'you can use'),
    ]
    cleaned = text
    for pat, rep in replacements:
        cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
    return cleaned


def is_conversational_greeting(query: str) -> bool:
    """Robust detection for conversational greetings, chit-chat, and polite expressions."""
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
        'feed', 'housing', 'pen', 'coop', 'vaccine', 'vaccination', 'egg', 'eggs', 'weight',
        'cost', 'expense', 'spent', 'buy', 'bought', 'naira', 'kudi', 'count', 'number'
    }

    has_greeting = bool(words.intersection(greeting_words)) or 'how far' in q or 'how are you' in q or 'how is work' in q or 'ina kwana' in q
    has_action = bool(words.intersection(agri_action_words))

    return has_greeting and not has_action


def handle_fast_intent(query: str, farm_id: str, language: str) -> Optional[str]:
    """Provides fast, 100% accurate grounded responses for greetings and farm inventory counts."""
    q = query.lower().strip()

    # 1. Robust Conversational Greetings Handler (0.00s, 100% natural, zero translation artifacts)
    if is_conversational_greeting(q):
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

    # 2. Authoritative Database Inventory Count Queries
    count_patterns = [
        r'\bhow many\b', r'\bnumber of animals\b', r'\bcount\b', r'\btotal animals\b',
        r'\blist animals\b', r'\blist all animals\b', r'\bshow animals\b', r'\banimal count\b',
        r'\banimals in the field\b', r'\banimals registered\b', r'\bdabbobi nawa\b', r'\bkaji nawa\b'
    ]
    if any(re.search(pat, q) for pat in count_patterns):
        import sqlite3
        from database import DB_PATH
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT id, name, species, breed FROM animals WHERE farm_id = ?", (farm_id,))
            animals = c.fetchall()
            farm = get_farm_by_id(farm_id)

        farm_name = farm["name"] if farm else "your farm"
        desc = farm["description"] if farm and farm.get("description") else ""
        count = len(animals)

        if language == "hausa":
            if count == 0:
                notes_part = f" Bayanin gonarku ya nuna: \"{desc}\", amma ba a yi wa kowace dabba rajista a rumbun bayanan ba tukuna." if desc else ""
                return f"Bisa ga bayanan gonarku ({farm_name}), a halin yanzu kuna da dabbobi 0 da aka yi wa rajista a rumbun bayanan.{notes_part}"
            else:
                animal_list = ", ".join([f"{a['id']} ({a['species']} - {a['name']})" for a in animals[:5]])
                return f"Bisa ga bayanan gonarku ({farm_name}), kuna da dabbobi {count} da aka yi wa rajista: {animal_list}."

        elif language == "pidgin":
            if count == 0:
                notes_part = f" Your farm profile write say: \"{desc}\", but you never register them inside the database." if desc else ""
                return f"According to your farm records ({farm_name}), you get 0 animals recorded for your database right now.{notes_part}"
            else:
                animal_list = ", ".join([f"{a['id']} ({a['species']} - {a['name']})" for a in animals[:5]])
                return f"According to your farm records ({farm_name}), you get {count} animals recorded: {animal_list}."

        else:
            if count == 0:
                notes_part = f" Your farm profile notes mention: \"{desc}\", but these have not been individually logged into the database yet." if desc else ""
                return f"According to your farm database records for {farm_name}, you currently have 0 registered animals.{notes_part}"
            else:
                animal_list = ", ".join([f"{a['id']} ({a['species']} - {a['name']})" for a in animals[:5]])
                return f"According to your farm database records for {farm_name}, you currently have {count} registered animals: {animal_list}."

    return None


def chat_completion(
    messages: List[Dict[str, str]],
    farm_id: str = "default_farm",
    thread_id: Optional[str] = None,
    language: str = "english"
) -> str:
    """Unified single-pass chat completion for FarmHand AI."""
    turn_start = time.time()
    norm_lang = normalize_language(language)
    print(f"\n[llm_engine] --- Chat Turn Start | Farm: '{farm_id}' | Lang: '{norm_lang}' ---")

    user_prompts = [m.get("content", "").strip() for m in messages if m.get("role") == "user"]
    last_user_query = user_prompts[-1] if user_prompts else ""

    # Step 1: Hausa -> English neural translation for inference
    effective_messages = []
    if norm_lang == "hausa":
        print(f"[llm_engine] Translating user input from Hausa to English: '{last_user_query}'")
        for m in messages:
            if m.get("role") == "user":
                translated_content = translate_ha_to_en(m.get("content", ""))
                effective_messages.append({"role": "user", "content": translated_content})
            else:
                effective_messages.append(m)
        current_query_en = effective_messages[-1]["content"] if effective_messages else last_user_query
    else:
        effective_messages = list(messages)
        current_query_en = last_user_query

    # Step 2: Instant Grounded Fast Intent Handler (< 0.001s for greetings and inventory)
    fast_ans = handle_fast_intent(current_query_en, farm_id=farm_id, language=norm_lang)
    if fast_ans:
        dt = time.time() - turn_start
        print(f"[llm_engine] Fast intent handler answered in {dt:.2f}s:\n{fast_ans}")
        print(f"[llm_engine] --- Chat Turn End ---\n")
        return fast_ans

    # Step 3: Fetch active farm profile to scope RAG search and advice
    farm = get_farm_by_id(farm_id)
    farm_species = farm["farm_type"] if farm and farm.get("farm_type") else "General"
    species_scope = farm_species if farm_species.lower() != "general" else ""

    # Step 4: Dynamic RAG retrieval (< 0.05s) with species scope
    rag_context = ""
    if len(current_query_en) > 3:
        search_query = f"{species_scope} {current_query_en}".strip() if species_scope else current_query_en
        rag_hits = search_knowledge_base(search_query, top_k=2)
        if rag_hits:
            snippets = [f"[{h.get('filename', 'Doc')}]: {h.get('text', '')[:300]}" for h in rag_hits]
            rag_context = "\n---\n".join(snippets)
            print(f"[llm_engine] Retrieved {len(rag_hits)} RAG context chunks for '{search_query}'.")

    # Step 5: Single-pass RAG synthesis
    llm = get_llm()
    if llm is None:
        return "[Fallback] Model not loaded."

    species_instruction = f"The active farm specializes in {farm_species}. Answer specifically for {farm_species} without asking the user what animals they have." if species_scope else "Answer the user's specific livestock or crop question directly."

    prompt = (
        "<|im_start|>system\n"
        "You are FarmHand AI, an expert agricultural and veterinary advisor.\n"
        f"{species_instruction}\n"
        "Provide practical, concise advice to the farmer based on the veterinary context below.\n\n"
        f"VETERINARY REFERENCE CONTEXT:\n{rag_context}\n<|im_end|>\n"
        f"<|im_start|>user\n{current_query_en}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "To help address this on your farm, "
    )

    print(f"[llm_engine] Running single-pass inference...")
    gen_start = time.time()
    response = llm(
        prompt,
        max_tokens=150,
        temperature=0.2,
        stop=["<|im_end|>", "<|im_start|>", "\n\nUser:", "Farmer:"]
    )
    raw_output = "To help address this on your farm, " + response["choices"][0]["text"].strip()
    gen_duration = time.time() - gen_start
    print(f"[llm_engine] Inference completed in {gen_duration:.2f}s | Output: {raw_output[:100]}...")

    # Step 6: Clean output & translate if Hausa
    final_output = raw_output
    if norm_lang == "english":
        final_output = convert_pidgin_to_clean_english(raw_output)
    elif norm_lang == "hausa":
        clean_en = convert_pidgin_to_clean_english(raw_output)
        print(f"[llm_engine] Translating cleaned English response to Hausa...")
        final_output = translate_en_to_ha(clean_en)

    total_time = time.time() - turn_start
    print(f"[llm_engine] TOTAL TURN TIME: {total_time:.2f}s")
    print(f"[llm_engine] --- Chat Turn End ---\n")
    return final_output