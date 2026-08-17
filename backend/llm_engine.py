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
_anti_json_logit_bias: Optional[Dict[int, float]] = None


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
    global _llm_instance, _anti_json_logit_bias
    if _llm_instance is None:
        if MODEL_PATH.exists():
            print(f"[llm_engine] Loading llama.cpp model from {MODEL_PATH} (threads={N_THREADS}, ctx={N_CTX})...")
            _llm_instance = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                verbose=False
            )
            # Token bias against '[' and '{' to prevent default tool-calling syntax
            sq_tokens = _llm_instance.tokenize(b'[')
            cu_tokens = _llm_instance.tokenize(b'{')
            _anti_json_logit_bias = {tok: -100.0 for tok in sq_tokens + cu_tokens}
        else:
            print(f"[llm_engine] Model file not found at {MODEL_PATH}")
            _llm_instance = None
    return _llm_instance


def clean_english_prose(text: str) -> str:
    """Transforms raw generated dialect tokens into polished, standard English prose."""
    phrases = [
        (r'\bna because of\b', 'This is likely caused by'),
        (r'\be-cause be\b', 'The cause may be'),
        (r'\bna likely\b', 'It is likely'),
        (r'\bna well-well\b', 'I understand'),
        (r'\bna properly\b', 'I understand'),
        (r'\bna only say\b', 'The primary causes are'),
        (r'\bna fit say\b', 'You can'),
        (r'\bpossible say\b', 'It is possible that'),
        (r'\bmake sure say\b', 'Ensure that'),
        (r'\bmake you check say\b', 'Please check that'),
        (r'\bmake you\b', 'please'),
        (r'\bmake we\b', 'let us'),
        (r'\bi go fit help you with\b', 'I can help you with'),
        (r'\bi go fit help you\b', 'I can help you'),
        (r'\bi go help you\b', 'I will help you'),
        (r'\bi go check\b', 'I will check'),
        (r'\bi dey help you with\b', 'I can assist you with'),
        (r'\bi dey here to help you\b', 'I am here to assist you'),
        (r'\bi dey well-well\b', 'I am doing well'),
        (r'\bwater dey flow\b', 'water flows'),
        (r'\bwater dey\b', 'water is'),
        (r'\bwetin you see\b', 'regarding your observation'),
        (r'\bwetin you suppose do\b', 'Recommended actions:'),
        (r'\bwetin dey happen\b', 'what is happening'),
        (r'\bwetin dey cause am\b', 'what causes it'),
        (r'\bwetin\b', 'what'),
        (r'\bdey cause am\b', 'causes it'),
        (r'\bdey cause\b', 'causes'),
        (r'\bdey come from\b', 'comes from'),
        (r'\bdey come\b', 'comes'),
        (r'\bdey enter for pen\b', 'in the pen'),
        (r'\bdey recorded for\b', 'recorded in'),
        (r'\bdey worse\b', 'get worse'),
        (r'\bworsen am\b', 'worsen the condition'),
        (r'\be dey worse\b', 'it gets worse'),
        (r'\be go worse\b', 'it will get worse'),
        (r'\be go\b', 'it will'),
        (r'\be be\b', 'it can be'),
        (r'\be fit be\b', 'it can be'),
        (r'\be dey\b', 'it is'),
        (r'\bwell-well\b', 'properly'),
        (r'\bwell-quick-quick\b', 'quickly'),
        (r'\bquick-quick\b', 'quickly'),
        (r'\bno fit\b', 'cannot'),
        (r'\bno dey\b', 'does not'),
        (r'\bno let\b', 'do not allow'),
        (r'\bno mixing\b', 'avoid mixing'),
        (r'\bno mix\b', 'do not mix'),
        (r'\bdey show say\b', 'shows that'),
        (r'\byou suppose say\b', 'you should verify if'),
        (r'\byou suppose\b', 'you should'),
        (r'\byou dey record say\b', 'you recorded that'),
        (r'\byou dey\b', 'you are'),
        (r'\byou get\b', 'you have'),
        (r'\byou fit\b', 'you can'),
        (r'\bif you get\b', 'if you have'),
        (r'\bif dem dey\b', 'if they are'),
        (r'\bdem dey\b', 'they are'),
        (r'\bdem be\b', 'it may be'),
        (r'\bfit trigger am\b', 'can trigger it'),
        (r'\bfit help\b', 'will help'),
        (r'\btrigger am\b', 'trigger it'),
        (r'\biver go use\b', 'you can use'),
        (r'\bsay say\b', 'what'),
        (r'\bwey dey\b', 'that is'),
        (r'\bna so\b', 'that is correct'),
        (r'\babeg\b', 'please'),
        (r'\bna\s+', 'This is '),
        (r'\bdey\s+', 'is '),
        (r'\bdi\s+', 'the '),
    ]
    cleaned = text
    for pat, rep in phrases:
        cleaned = re.sub(pat, rep, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


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
        'cost', 'expense', 'spent', 'buy', 'bought', 'naira', 'kudi', 'count', 'number', 'many'
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

    # Step 2: Instant Grounded Fast Intent Handler strictly for greetings (< 0.001s)
    fast_ans = handle_fast_intent(current_query_en, language=norm_lang)
    if fast_ans:
        dt = time.time() - turn_start
        print(f"[llm_engine] Fast greeting handler answered in {dt:.2f}s:\n{fast_ans}")
        print(f"[llm_engine] --- Chat Turn End ---\n")
        return fast_ans

    # Step 3: Fetch active farm database profile & context
    farm_summary = get_system_context_summary(farm_id)
    farm = get_farm_by_id(farm_id)
    farm_species = farm["farm_type"] if farm and farm.get("farm_type") else "General"
    species_scope = farm_species if farm_species.lower() != "general" else ""

    # Step 4: Selective High-Relevance RAG Retrieval
    rag_context = ""
    if len(current_query_en) > 5:
        rag_hits = search_knowledge_base(f"{species_scope} {current_query_en}".strip(), top_k=2)
        filtered_snippets = []
        irrelevant_terms = ["gender analysis", "ex-ante", "rendille", "theileria", "macro-economic"]
        for h in rag_hits:
            text = h.get("text", "")
            filename = h.get("filename", "").lower()
            if not any(term in text.lower() or term in filename for term in irrelevant_terms):
                if len(text) > 50:
                    filtered_snippets.append(f"[{h.get('filename', 'Ref')}]: {text[:250]}")
        if filtered_snippets:
            rag_context = "\n---\n".join(filtered_snippets)
            print(f"[llm_engine] High-relevance RAG context chunks: {len(filtered_snippets)}")

    # Step 5: Load LLM with Logit Bias
    llm = get_llm()
    if llm is None:
        return "[Fallback] Model not loaded."

    knowledge_section = f"\nVETERINARY REFERENCE CONTEXT:\n{rag_context}\n" if rag_context.strip() else ""

    if norm_lang == "pidgin":
        lang_instruction = "Answer the farmer directly in helpful, warm Nigerian Pidgin with practical clinical causes, first aid, and prevention steps."
    else:
        lang_instruction = "Answer the farmer directly with practical clinical causes, first aid, and prevention steps in standard English."

    system_prompt = (
        "You are FarmHand AI, an expert agricultural and veterinary advisor for farmers.\n"
        f"{farm_summary}\n"
        f"{lang_instruction}\n"
        "When the farmer asks about their farm, animals, or notes, cite the ACTIVE FARM PROFILE above.\n"
        "Do NOT change the subject. Do NOT output random numbers, JSON brackets, or code."
        f"{knowledge_section}"
    )

    # Build ChatML prompt with multi-turn conversation history
    prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    for m in effective_messages[-6:]:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if content:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prompt_parts.append(f"<|im_start|>assistant\n")
    full_prompt = "\n".join(prompt_parts)

    print(f"[llm_engine] Running multi-turn inference...")
    gen_start = time.time()
    response = llm(
        full_prompt,
        max_tokens=150,
        temperature=0.1,
        logit_bias=_anti_json_logit_bias,
        stop=["<|im_end|>", "<|im_start|>", "\n\nUser:", "Farmer:"]
    )
    raw_output = response["choices"][0]["text"].strip()
    gen_duration = time.time() - gen_start
    print(f"[llm_engine] Inference completed in {gen_duration:.2f}s | Output: {raw_output[:100]}...")

    # Step 6: Post-process & Language formatting
    if norm_lang == "pidgin":
        final_output = raw_output  # Authentic Pidgin preserved!
    elif norm_lang == "hausa":
        clean_en = clean_english_prose(raw_output)
        print(f"[llm_engine] Translating cleaned response to Hausa...")
        ha_translated = translate_en_to_ha(clean_en)
        religious_artifacts = ["littafi mai tsarki", "ãdalci", "sikẽlin", "la'ĩmi", "al'ummai", "karin magana"]
        if any(art in ha_translated.lower() for art in religious_artifacts):
            print(f"[llm_engine] Detected MarianMT religious artifact in translation. Using clean advisory format.")
            final_output = f"Shawarar FarmHand: {clean_en}"
        else:
            final_output = ha_translated
    else:  # English
        final_output = clean_english_prose(raw_output)

    total_time = time.time() - turn_start
    print(f"[llm_engine] TOTAL TURN TIME: {total_time:.2f}s")
    print(f"[llm_engine] --- Chat Turn End ---\n")
    return final_output