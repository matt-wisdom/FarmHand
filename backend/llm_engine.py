import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar

from database import get_system_context_summary
from rag_pipeline import search_knowledge_base
from tool_registry import TOOL_MAP, execute_tool
from translator import translate_en_to_ha, translate_ha_to_en

logger = logging.getLogger("FarmHandEngine")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "qwen2.5-3b-instruct.Q4_K_M.gguf"

# Hardware-scaled CPU threads and full 4k context window
N_CTX = 4096
N_THREADS = max(1, os.cpu_count() or 4)

# --- FIX A: minimum similarity score required to trust a RAG hit ---
# Tune this against your embedding model's actual score distribution. If your
# search_knowledge_base returns cosine similarity in [0,1], 0.55 is a reasonable
# starting floor; if it returns raw distances (lower = better), invert this logic.
RAG_MIN_SCORE = 0.55

_llm_instance: Llama | None = None
_llama_grammar_instance: LlamaGrammar | None = None
_anti_json_logit_bias: dict[int, float] | None = None
_english_logit_bias: dict[int, float] | None = None


def normalize_language(lang: str | None) -> str:
    """Normalize language code/name to 'english', 'hausa', or 'pidgin'."""
    if not lang:
        return "english"
    cleaned = str(lang).strip().lower()
    if cleaned in ["ha", "hausa"] or cleaned.startswith("ha-") or "hausa" in cleaned:
        return "hausa"
    if (
        cleaned in ["pg", "pidgin", "pcm"]
        or cleaned.startswith("pid")
        or "pidgin" in cleaned
    ):
        return "pidgin"
    return "english"


def build_tools_json_schema() -> dict[str, Any]:
    tool_names = list(TOOL_MAP.keys())
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "enum": tool_names},
                "arguments": {"type": "object"},
            },
            "required": ["function_name", "arguments"],
            "additionalProperties": False,
        },
    }


def get_routing_system_prompt(farm_id: str = "default_farm") -> dict[str, str]:
    """Universal routing prompt with strict animal acquisition vs financial expense fidelity and active farm grounding."""
    from database import get_farm_by_id, normalize_species_name

    farm = get_farm_by_id(farm_id)
    farm_name = farm.get("name", "General Farm") if farm else "General Farm"
    farm_type = farm.get("farm_type", "General") if farm else "General"
    farm_desc = farm.get("description", "") if farm else ""

    norm_species = ""
    if farm_type and farm_type.lower() not in ("general", "other", "mixed"):
        norm_species = normalize_species_name(farm_type).lower()

    farm_context_header = (
        "ACTIVE FARM SCOPE & IDENTITY:\n"
        f"- Active Farm: {farm_name}\n"
        f"- Target Farm Species: {farm_type}"
        + (f" (Canonical tool species: '{norm_species}')\n" if norm_species else "\n")
        + (f"- Farm Profile Notes: {farm_desc}\n" if farm_desc else "")
        + f"- CRITICAL SCOPE RULE: This farm is a {farm_type} farm. Default all ambiguous flock registrations, inventory checks, feed formulations, observations, and veterinary search queries to the active farm's species ('{norm_species or farm_type.lower()}'). If the farmer explicitly specifies another animal by name, always respect that requested species.\n\n"
    )

    return {
        "role": "system",
        "content": (
            "You are the tool routing engine for FarmHand AI.\n"
            "Output ONLY a valid JSON array with the single best tool call.\n\n"
            f"{farm_context_header}"
            "CRITICAL ROUTING RULES:\n"
            "1. ANIMAL ACQUISITIONS & FLOCK HEADCOUNTS (register_flock, list_animals):\n"
            "   - When the farmer mentions acquiring, buying, adding, hatching, or owning animals (e.g. 'I just bought 20', 'Just acquired 200 fingerlings', 'bought 10', 'We currently have 9', 'we hatched 50 chicks', '10 died'):\n"
            "     ALWAYS route to register_flock with event_type='purchase' (or 'initial_count', 'birth', 'mortality') and count matching the number of animals.\n"
            f"     If the animal species is omitted or implied, use the active farm's species ('{norm_species or farm_type.lower()}').\n"
            "     NEVER route animal headcount acquisitions to write_expenditure unless an explicit currency amount was provided by the farmer.\n"
            "2. OPERATING EXPENSES & FINANCIAL SPENDING (write_expenditure, list_expenditures):\n"
            "   - Route to write_expenditure ONLY when the farmer explicitly provides a monetary cost, price, or spending amount (e.g. 'Spent 15,000 NGN on feed', 'Paid 5000 for antibiotics', 'bought feed for 18000').\n"
            "   - NEVER fabricate or invent financial amounts if none were provided.\n"
            "3. CLINICAL SYMPTOMS & VETERINARY ADVICE (query_knowledge_base):\n"
            "   - When the farmer describes illness, symptoms, or disease, or asks for clinical guidance (e.g. 'foaming from mouth', 'coughing', 'green diarrhea', 'what is coccidiosis', 'how to treat disease'):\n"
            f"     ALWAYS route to query_knowledge_base with a targeted veterinary search query matching the symptoms and the active farm's species ('{norm_species or farm_type.lower()}').\n"
            "4. FEED RECIPES & MIXTURES (optimize_feed_formulation):\n"
            "   - When the farmer asks for a feed formula, recipe, or cheapest feed mix (e.g. 'mix cheap 100kg broiler feed', 'formulate layer mash', 'catfish feed mix'):\n"
            "     ALWAYS route to optimize_feed_formulation.\n"
            "5. STRICT SPECIES FIDELITY:\n"
            "   - chickens / hens / broilers / layers / chicks / birds -> 'poultry'\n"
            "   - goats / kids / bucks / does -> 'goat'\n"
            "   - cows / cattle / bulls / calves / heifers -> 'cattle'\n"
            "   - sheep / rams / ewes / lambs -> 'sheep'\n"
            "   - pigs / swine / piglets -> 'pig'\n"
            "   - fish / catfish / tilapia / fingerlings -> 'fish'\n\n"
            "TOOLS:\n"
            "- list_animals(species: str, date_str: str): Check animal headcount or how many animals are on the farm.\n"
            "- register_flock(species: str, count: int, event_type: str, notes: str): Record animals (purchase, birth, initial_count, mortality, sale).\n"
            "- list_expenditures(category: str): View recorded farm spending.\n"
            "- write_expenditure(category: str, amount: float, description: str): Record a monetary expense where the farmer explicitly gave an amount.\n"
            "- log_farm_observation(species: str, observation: str, category: str): Record background setup memory about farm infrastructure or equipment.\n"
            "- optimize_feed_formulation(target_profile: str, batch_size_kg: float): Calculate least-cost feed formula using Linear Programming.\n"
            "- query_knowledge_base(search_query: str): Search veterinary manuals for illness, symptoms, treatment, drugs, or care guidance.\n\n"
            "EXAMPLES:\n"
            'Farmer: \'I just bought 20 chickens\' -> [{"function_name": "register_flock", "arguments": {"species": "poultry", "count": 20, "event_type": "purchase", "notes": "bought 20 chickens"}}]\n'
            'Farmer: \'Just acquired 200 fingerlings of tilapia\' -> [{"function_name": "register_flock", "arguments": {"species": "fish", "count": 200, "event_type": "purchase", "notes": "acquired 200 fingerlings of tilapia"}}]\n'
            'Farmer: \'bought 10 cows\' -> [{"function_name": "register_flock", "arguments": {"species": "cattle", "count": 10, "event_type": "purchase", "notes": "bought 10 cows"}}]\n'
            'Farmer: \'We currently have 9 goats\' -> [{"function_name": "register_flock", "arguments": {"species": "goat", "count": 9, "event_type": "initial_count", "notes": "initial count"}}]\n'
            'Farmer: \'3 chickens died this morning\' -> [{"function_name": "register_flock", "arguments": {"species": "poultry", "count": 3, "event_type": "mortality", "notes": "3 chickens died"}}]\n'
            'Farmer: \'Spent 18,000 on feed\' -> [{"function_name": "write_expenditure", "arguments": {"category": "feed", "amount": 18000, "description": "feed purchase"}}]\n'
            'Farmer: \'1 animal suddenly died this morning. It was foaming from the mouth\' -> [{"function_name": "query_knowledge_base", "arguments": {"search_query": "'
            + (norm_species or "poultry")
            + ' sudden death foaming mouth poisoning symptoms emergency care"}}]\n'
            'Farmer: \'how to treat coccidiosis\' -> [{"function_name": "query_knowledge_base", "arguments": {"search_query": "'
            + (norm_species or "poultry")
            + ' coccidiosis treatment medication prevention"}}]\n'
            'Farmer: \'How can I mix cheap 100kg feed for my 3-week broilers\' -> [{"function_name": "optimize_feed_formulation", "arguments": {"target_profile": "broiler_starter", "batch_size_kg": 100}}]\n'
            'Farmer: \'how many animals do i have\' -> [{"function_name": "list_animals", "arguments": {"species": "'
            + (norm_species or "poultry")
            + '"}}]\n'
        ),
    }


def get_llama_grammar() -> LlamaGrammar | None:
    global _llama_grammar_instance
    if _llama_grammar_instance is None:
        schema = build_tools_json_schema()
        schema_json = json.dumps(schema)
        _llama_grammar_instance = LlamaGrammar.from_json_schema(schema_json)
    return _llama_grammar_instance


def get_llm() -> Llama | None:
    """Lazy loader for llama.cpp model instance."""
    global _llm_instance, _anti_json_logit_bias, _english_logit_bias
    if _llm_instance is None:
        if MODEL_PATH.exists():
            print(
                f"[llm_engine] Loading llama.cpp model from {MODEL_PATH} (threads={N_THREADS}, ctx={N_CTX})..."
            )
            _llm_instance = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_threads_batch=N_THREADS,
                n_batch=512,
                chat_format="chatml",
                verbose=False,
            )
            # Comprehensive anti-JSON token bias to prevent tool-calling output in Pass 3
            json_symbols = [
                "[",
                " [",
                "\n[",
                "{",
                " {",
                "\n{",
                '{"',
                "function",
                "function_name",
                '"function_name"',
                "write_",
                "list_",
                "register_",
            ]
            _anti_json_logit_bias = {}
            for s in json_symbols:
                for tok in _llm_instance.tokenize(s.encode("utf-8"), add_bos=False):
                    _anti_json_logit_bias[tok] = -100.0

            # English-mode bias: suppress JSON tokens + Pidgin particles directly in llama.cpp
            pidgin_words = [
                "dey",
                " dey",
                "Dey",
                " Dey",
                "\ndey",
                "\nDey",
                "wey",
                " wey",
                "Wey",
                " Wey",
                "\nwey",
                "\nWey",
                "wetin",
                " wetin",
                "Wetin",
                " Wetin",
                "una",
                " una",
                "Una",
                " Una",
                "sabi",
                " sabi",
                "Sabi",
                " Sabi",
                "abeg",
                " abeg",
                "Abeg",
                " Abeg",
                "oga",
                " oga",
                "Oga",
                " Oga",
                "sef",
                " sef",
                "Sef",
                " Sef",
                "wahala",
                " wahala",
                "Wahala",
                " Wahala",
                "shey",
                " shey",
                "Shey",
                " Shey",
                "kuku",
                " kuku",
                "Kuku",
                " Kuku",
                "comot",
                " comot",
                "commot",
                " commot",
                "chop",
                " chop",
                "Chop",
                " Chop",
                "pikin",
                " pikin",
                "Pikin",
                " Pikin",
                "nau",
                " nau",
                "Nau",
                " Nau",
                "oya",
                " oya",
                "Oya",
                " Oya",
                "abi",
                " abi",
                "Abi",
                " Abi",
                "well-well",
                " well-well",
                "quick-quick",
                " quick-quick",
                "small-small",
                " small-small",
                "don",
                " don",
                "Don",
                " Don",
                "dem",
                " dem",
                "Dem",
                " Dem",
                "\ndem",
                "\nDem",
                "di",
                " di",
                "Di",
                " Di",
                "\ndi",
                "\nDi",
                " am",
                "\nam",
            ]
            _english_logit_bias = dict(_anti_json_logit_bias)
            for w in pidgin_words:
                toks = _llm_instance.tokenize(w.encode("utf-8"), add_bos=False)
                for t in toks:
                    _english_logit_bias[t] = -100.0
        else:
            print(f"[llm_engine] Model file not found at {MODEL_PATH}")
            _llm_instance = None
    return _llm_instance


def parse_tool_calls(output_text: str) -> tuple[bool, list[dict[str, Any]]]:
    clean_text = output_text.strip()
    if not (clean_text.startswith("[") and "function_name" in clean_text):
        return False, []

    try:
        data = json.loads(clean_text)
        if isinstance(data, list) and len(data) > 0:
            valid_calls = []
            for item in data:
                if (
                    isinstance(item, dict)
                    and "function_name" in item
                    and item["function_name"] in TOOL_MAP
                ):
                    args = item.get("arguments", {})
                    if not args:
                        args = {k: v for k, v in item.items() if k != "function_name"}
                    valid_calls.append(
                        {"function_name": item["function_name"], "arguments": args}
                    )
            if valid_calls:
                return True, valid_calls
    except Exception as e:
        logger.debug("Error parsing tool calls: %s", e)
    return False, []


def is_conversational_greeting(query: str) -> bool:
    """Strictly checks if a short message is purely a greeting, pleasantry, or thank-you expression."""
    clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if not clean:
        return True

    # Exact common greeting and pleasantry phrases
    exact_greetings = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you doing",
        "how is work",
        "howfar",
        "how far",
        "sup",
        "yo",
        "welcome",
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
        "thank you very much",
        "sannu",
        "sannu da aiki",
        "barka",
        "barka da asuba",
        "barka da rana",
        "barka da yamma",
        "ina kwana",
        "ina wuni",
        "yaya dai",
        "nagode",
        "na gode",
        "nagode sosai",
        "na gode sosai",
    }
    if clean in exact_greetings:
        return True

    # Short 1-2 word combinations where every word is a greeting token
    words = clean.split()
    greeting_tokens = {
        "hello",
        "hi",
        "hey",
        "sannu",
        "barka",
        "morning",
        "afternoon",
        "evening",
        "welcome",
        "thanks",
        "thank",
        "nagode",
        "gode",
        "howfar",
        "sup",
        "yo",
        "kwana",
        "wuni",
    }
    return len(words) <= 2 and all(w in greeting_tokens for w in words)


def handle_fast_intent(query: str, language: str) -> str | None:
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


def format_database_tool_context(tool_results: list[dict[str, Any]]) -> str:
    lines = []
    for tr in tool_results:
        tool = tr["tool"]
        res = tr["result"]

        if tool == "list_animals" and isinstance(res, dict):
            data = res.get("data", [])
            total = res.get(
                "total",
                sum(a.get("count", 0) for a in data) if isinstance(data, list) else 0,
            )
            lines.append("FARM FLOCK INVENTORY QUERY RESULT:")
            if isinstance(data, dict):
                # Historical date query
                lines.append(
                    f"- Historical Count on {data.get('as_of_date')}: {data.get('count', data.get('total_flock_size', 0))} {data.get('species', 'animals')}"
                )
            elif data:
                flock_strs = [
                    f"{a.get('species', 'Unknown').capitalize()}: {a.get('count', 0)}"
                    for a in data
                ]
                lines.append(
                    f"- Current Flock Counts: {', '.join(flock_strs)} (Total: {total})"
                )
            else:
                lines.append(
                    "- Current Inventory: None (0 animals recorded in database)"
                )

        elif tool == "register_flock" and isinstance(res, dict):
            lines.append("FLOCK REGISTRATION RESULT:")
            if res.get("status") == "success":
                entry = res.get("entry", {})
                lines.append(
                    f"- Successfully recorded {entry.get('event_type')} of {entry.get('count_change', 0):+d} {entry.get('species')}. New total flock balance is {entry.get('new_total', 0)}."
                )
            else:
                lines.append(
                    f"- Failed to record flock: {res.get('message', 'Unknown error')}"
                )

        elif tool == "list_expenditures" and isinstance(res, dict):
            cnt = res.get("count", 0)
            data = res.get("data", [])
            lines.append("FARM EXPENDITURES QUERY RESULT:")
            lines.append(f"- Total Recorded Expenditures: {cnt}")
            if data:
                exp_strs = [
                    f"ID {e.get('id')}: NGN {e.get('amount', 0):,.2f} for {e.get('category')} - {e.get('description')}"
                    for e in data[:5]
                ]
                lines.append(f"- Recent Expenses: {'; '.join(exp_strs)}")
            else:
                lines.append("- Recent Expenses: None recorded")

        elif tool == "write_expenditure" and isinstance(res, dict):
            d = res.get("data", {})
            lines.append("EXPENDITURE RECORDED RESULT:")
            lines.append(
                f"- Successfully recorded expense of NGN {d.get('amount', 0):,.2f} under '{d.get('category')}' ({d.get('description', '')})."
            )

        elif tool == "list_health_logs" and isinstance(res, dict):
            cnt = res.get("count", 0)
            data = res.get("data", [])
            lines.append("FLOCK HEALTH LOGS QUERY RESULT:")
            lines.append(f"- Total Health Logs: {cnt}")
            if data:
                log_strs = [
                    f"Flock ({h.get('species')}): {h.get('event_type')} - {h.get('notes')}"
                    for h in data[:5]
                ]
                lines.append(f"- Health History: {'; '.join(log_strs)}")
            else:
                lines.append("- Health History: None recorded")

        elif tool == "write_health_log" and isinstance(res, dict):
            d = res.get("data", {})
            lines.append("HEALTH LOG RECORDED RESULT:")
            lines.append(
                f"- Successfully logged health event '{d.get('event_type')}' for animal {d.get('animal_id')} ({d.get('notes', '')})."
            )

        elif tool == "log_farm_observation" and isinstance(res, dict):
            mem = res.get("memory", {})
            lines.append("FARM CLINICAL OBSERVATION LOGGED:")
            lines.append(
                f"- Saved observation for {mem.get('species', 'Flock')} ({mem.get('category', 'symptom')}): {mem.get('observation')}"
            )

        elif tool == "optimize_feed_formulation" and isinstance(res, dict):
            f = res.get("formulation", {})
            if f.get("success"):
                lines.append("OPTIMIZED FEED FORMULATION RESULT:")
                lines.append(
                    f"- Target Diet: {f.get('target_display_name')} ({f.get('batch_size_kg')} kg Batch)"
                )
                lines.append(
                    f"- Cost per 50kg Bag: NGN {f.get('cost_50kg_bag', 0):,.2f} (NGN {f.get('cost_per_kg', 0):,.2f} / kg, {f.get('savings_percentage', 0)}% savings vs commercial feed)"
                )
                recipe_items = [
                    f"{item['name']}: {item['weight_kg']} kg ({item['proportion_percent']}%)"
                    for item in f.get("recipe", [])
                ]
                lines.append(f"- Ingredients Breakdown: {', '.join(recipe_items)}")
                lines.append(
                    f"- Mixing Steps: {'; '.join(f.get('mixing_instructions', []))}"
                )
            else:
                lines.append(
                    f"- Feed Optimization Failed: {f.get('message', 'Infeasible formulation')}"
                )

        else:
            lines.append(f"- {tool}: {json.dumps(res)}")

    return "\n".join(lines)


# --- FIX A helper: filter RAG hits by a relevance floor before trusting them ---
def filter_relevant_rag_hits(
    rag_hits: list[dict[str, Any]], min_score: float = RAG_MIN_SCORE
) -> list[dict[str, Any]]:
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
        # reliably; keep the hit but log it so this gets noticed and fixed upstream.
        if score is None:
            print(
                f"[llm_engine] WARNING: RAG hit for '{h.get('filename')}' has no 'score' field; "
                f"cannot apply relevance floor. Consider updating search_knowledge_base to return scores."
            )
            filtered.append(h)
            continue
        if score >= min_score:
            filtered.append(h)
        else:
            print(
                f"[llm_engine] Dropping low-relevance RAG hit '{h.get('filename')}' (score={score:.3f} < {min_score})"
            )

    return filtered


def generate_stateless_answer(
    llm: Llama,
    context_data: str,
    user_question: str,
    norm_lang: str,
    db_summary: str,
    messages: list[dict[str, str]] | None = None,
    stream: bool = False,
):
    """Pass 3: Natural language synthesis using strictly positive prompting and API Chat Wrapper."""

    # Budget context length to prevent exceeding token limits
    safe_context = (
        context_data[:2500].strip()
        if context_data
        else "No additional reference documents found."
    )

    if norm_lang == "pidgin":
        system_prompt = (
            "You are FarmHand AI, an expert agricultural, livestock, and aquaculture advisor.\n"
            "Respond to the farmer in warm, helpful, natural Pidgin English sentences.\n\n"
            f"FARM INVENTORY & PROFILE:\n{db_summary}\n\n"
            f"REFERENCE KNOWLEDGE BASE & CLINICAL RECORDS:\n{safe_context}\n\n"
            "INSTRUCTIONS:\n"
            "- Directly and accurately answer the farmer's question or acknowledge ledger entries.\n"
            "- GROUNDING IN ACTIVE FARM SCOPE & LEDGER ACTIONS:\n"
            "  * Follow the active farm species scope shown above. When the species is not specified by the farmer, tailor advice to the active farm's species. When the farmer names a specific animal (e.g. goats, chickens, cattle, fish, pigs, sheep), always provide advice for that requested animal.\n"
            "  * If REFERENCE KNOWLEDGE BASE has FLOCK REGISTRATION RESULT or EXPENDITURE RECORDED RESULT:\n"
            "    1. Confirm that the entry is recorded in the ledger (e.g. 'I don record the 20 new chickens inside your flock ledger successfully!').\n"
            "    2. Provide 2-3 quick, high-value initial care tips for the animals (e.g. clean warm pen, anti-stress water, starter feed).\n"
            "    3. Do NOT invent unprompted disease diagnoses or jump to unrelated animals when the farmer is just adding new animals.\n"
            "- For feeding & nutrition: explain ingredients, protein needs, local substitutes (e.g. rice bran, fish meal), and preparation steps.\n"
            "- For diseases & symptoms: evaluate symptoms against reference documents, name suspected conditions (e.g. PPR, enterotoxemia/pulpy kidney, plant/chemical poisoning), give supportive first aid, and advise calling a vet.\n"
            "- NEVER output placeholder promises like 'I will log this now', 'I am logging...', 'I go log am now', or 'I go help you record...'. Provide immediate confirmation or clinical advice directly."
        )
        prefill = ""

        pidgin_few_shot_1_user = (
            "My goat dey cough well-well and fever dey hold am with running nose."
        )
        pidgin_few_shot_1_assistant = (
            "Possible cause na contagious viral sickness like PPR (Peste des Petits Ruminants) or acute respiratory infection.\n\n"
            "Wetin you suppose do quick-quick:\n"
            "1. Isolate the sick goat immediately make e no infect other goats for pen.\n"
            "2. Keep the pen warm, dry, and provide clean water.\n"
            "3. Call veterinary doctor make them give proper antibiotics and supportive care."
        )

        pidgin_few_shot_2_user = "How I fit mix feed for broiler?"
        pidgin_few_shot_2_assistant = (
            "To mix good broiler feed:\n"
            "1. Mix maize or guinea corn (50-55%) with roasted soybean meal or fish meal (25-30%) for high protein.\n"
            "2. Add wheat offal or rice bran (10-15%) to give energy and fiber.\n"
            "3. Add bone meal (2-3%), limestone (1%), salt, and broiler premix."
        )

        pidgin_few_shot_3_user = "Hello, I just bought 20 chickens"
        pidgin_few_shot_3_assistant = (
            "I don record the 20 new chickens inside your flock ledger successfully! Your total bird count don update now.\n\n"
            "**Quick advice for the new birds**:\n"
            "1. **Warm Pen**: Put them for clean, warm pen with fresh wood shavings make cold no catch them.\n"
            "2. **Clean Water**: Give them clean water with anti-stress vitamins make them relax after transport.\n"
            "3. **Good Feed**: Give them quality chick starter feed make them grow quick-quick."
        )

        chatml_parts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>",
            f"<|im_start|>user\n{pidgin_few_shot_1_user}<|im_end|>",
            f"<|im_start|>assistant\n{pidgin_few_shot_1_assistant}<|im_end|>",
            f"<|im_start|>user\n{pidgin_few_shot_2_user}<|im_end|>",
            f"<|im_start|>assistant\n{pidgin_few_shot_2_assistant}<|im_end|>",
            f"<|im_start|>user\n{pidgin_few_shot_3_user}<|im_end|>",
            f"<|im_start|>assistant\n{pidgin_few_shot_3_assistant}<|im_end|>",
        ]
    else:
        system_prompt = (
            "You are FarmHand AI, an expert agricultural, livestock, and aquaculture specialist.\n"
            "Language: Standard International English.\n"
            "Rule: Write ONLY in clear, grammatically correct standard English. Never use slang or colloquial particles.\n\n"
            f"FARM INVENTORY & PROFILE:\n{db_summary}\n\n"
            f"REFERENCE KNOWLEDGE BASE & CLINICAL RECORDS:\n{safe_context}\n\n"
            "INSTRUCTIONS:\n"
            "- Directly and accurately answer the farmer's specific question or confirm ledger updates in natural, clear sentences.\n"
            "- GROUNDING IN ACTIVE FARM SCOPE & LEDGER ACTIONS:\n"
            "  * Tailor responses to the target species of the active farm profile shown above when the user does not specify an animal. When the farmer explicitly mentions a specific animal (e.g. goats, chickens, cattle, fish, pigs, sheep), always answer for that requested animal.\n"
            "  * When REFERENCE KNOWLEDGE BASE contains a FLOCK REGISTRATION RESULT or EXPENDITURE RECORDED RESULT:\n"
            "    1. Confirm the transaction clearly (e.g. 'Successfully recorded the purchase of 20 chickens (Poultry) into your flock ledger. Your total flock balance is now updated.').\n"
            "    2. Provide 2-3 concise, practical next steps (e.g. brooding warmth, clean water with anti-stress vitamins, starter feed).\n"
            "    3. Do NOT invent unprompted disease symptoms or jump to unrelated animal species when acknowledging a standard animal purchase or headcount addition.\n"
            "    4. NEVER say 'I can record this if you like' when the record is already logged in the database.\n"
            "- For disease names, overviews, or 'what is' questions (e.g. 'tetanus in goats', 'what is tetanus', 'coccidiosis in poultry'): Directly provide a comprehensive, accurate explanation. Always state what the disease is, its causative pathogen (bacteria, virus, or parasite), mode of transmission/infection (e.g. soil bacteria entering puncture wounds), key clinical symptoms (e.g. muscle stiffness, lockjaw, rigid posture), and prevention/control before veterinary care.\n"
            "- For clinical symptoms / sudden death queries (e.g. sudden mortality, foaming from mouth, respiratory distress, scours): Identify the probable suspect diseases based on reference documents (e.g. Enterotoxemia / Pulpy Kidney, acute poisoning/chemical toxin, or acute PPR), explain why, and provide immediate biosecurity and supportive action steps.\n"
            "- NEVER ask questions about database tags, pen numbers, or technical IDs during a medical/symptom inquiry. Focus directly on clinical triage and supportive care.\n"
            "- For feeding, formulations, recipes, or multi-step procedures: Detail practical ingredients, crude protein percentages, local substitutes, and preparation steps clearly.\n"
            "- If reference excerpts focus on a specific procedure (e.g. castration or wound care), still provide the full, direct answer to the farmer's disease question.\n"
            "- Do NOT output JSON or function calls.\n"
            "- NEVER output placeholder promises (e.g. do NOT say 'I will log this now', 'I am recording this now', 'I go help you...', or 'I will guide you through the steps'). Provide your complete diagnostic response immediately."
        )
        prefill = ""

        few_shot_1_user = (
            "What causes sudden shivering and green diarrhea in young calves?"
        )
        few_shot_1_assistant = (
            "Sudden shivering and profuse watery green diarrhea in young calves typically indicate acute colibacillosis (E. coli scours) or acute coccidiosis.\n\n"
            "**Immediate Action & Management**:\n"
            "1. **Rehydration**: Administer warm oral electrolyte rehydration solutions immediately to prevent lethal dehydration.\n"
            "2. **Isolation & Warmth**: Move affected calves to a clean, dry, disinfected pen with fresh bedding.\n"
            "3. **Veterinary Care**: Consult your veterinarian for appropriate antimicrobial therapy and fluid management."
        )

        few_shot_2_user = "How can I formulate low-cost grower feed for broilers?"
        few_shot_2_assistant = (
            "To formulate a low-cost broiler grower feed:\n"
            "1. Combine local energy sources such as maize or sorghum (50-55%) with protein sources like roasted soybean meal or fish meal (25-30%).\n"
            "2. Add wheat offal or rice bran (10-15%) for fiber and energy balance.\n"
            "3. Mix in bone meal (2-3%), limestone (1%), salt (0.3%), and a standard broiler premix with essential vitamins and minerals."
        )

        few_shot_3_user = "Hello, I just bought 20 chickens"
        few_shot_3_assistant = (
            "Successfully recorded the purchase of 20 chickens (Poultry) into your flock ledger! Your active flock balance is now updated.\n\n"
            "**Recommended Initial Management Steps**:\n"
            "1. **Brooding & Warmth**: Ensure the coop is clean, dry, draft-free, and pre-warmed with fresh litter.\n"
            "2. **Hydration & Anti-Stress**: Provide clean drinking water mixed with vital electrolytes or glucose for the first 24-48 hours.\n"
            "3. **Nutrition**: Supply high-protein chick starter mash (20-22% crude protein) in easily accessible feeders."
        )

        chatml_parts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>",
            f"<|im_start|>user\n{few_shot_1_user}<|im_end|>",
            f"<|im_start|>assistant\n{few_shot_1_assistant}<|im_end|>",
            f"<|im_start|>user\n{few_shot_2_user}<|im_end|>",
            f"<|im_start|>assistant\n{few_shot_2_assistant}<|im_end|>",
            f"<|im_start|>user\n{few_shot_3_user}<|im_end|>",
            f"<|im_start|>assistant\n{few_shot_3_assistant}<|im_end|>",
        ]

    # Build ChatML history
    clean_user_question = user_question.strip()
    if clean_user_question.startswith("Farmer:"):
        clean_user_question = (
            clean_user_question.replace("Farmer:", "")
            .strip()
            .strip("'")
            .strip('"')
            .strip()
        )

    if messages and len(messages) > 1:
        for m in messages[-4:-1]:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if content:
                chatml_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        last_user_content = (messages[-1].get("content") or clean_user_question).strip()
        chatml_parts.append(f"<|im_start|>user\n{last_user_content}<|im_end|>")
    else:
        chatml_parts.append(f"<|im_start|>user\n{clean_user_question}<|im_end|>")

    chatml_parts.append(f"<|im_start|>assistant\n{prefill}")
    raw_prompt = "\n".join(chatml_parts)

    # Calculate token headroom dynamically
    est_prompt_tokens = len(raw_prompt) // 3
    max_gen_tokens = max(128, min(450, N_CTX - est_prompt_tokens - 100))

    bias = _english_logit_bias if norm_lang != "pidgin" else _anti_json_logit_bias

    if stream:
        response_stream = llm.create_completion(
            prompt=raw_prompt,
            max_tokens=max_gen_tokens,
            temperature=0.1,
            repeat_penalty=1.15,
            logit_bias=bias,
            stop=["<|im_end|>", "<|im_start|>"],
            stream=True,
        )

        def token_generator():
            for chunk in response_stream:
                text = chunk["choices"][0]["text"]
                if text:
                    yield text

        return token_generator()

    response = llm.create_completion(
        prompt=raw_prompt,
        max_tokens=max_gen_tokens,
        temperature=0.1,
        repeat_penalty=1.15,
        logit_bias=bias,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    generated_text = response["choices"][0]["text"].strip()
    content = f"{prefill}{generated_text}".strip()
    return content


def chat_completion(
    messages: list[dict[str, str]],
    farm_id: str = "default_farm",
    thread_id: str | None = None,
    language: str = "english",
) -> str:
    """Unified multi-turn chat completion for FarmHand AI."""
    turn_start = time.time()
    norm_lang = normalize_language(language)
    print(
        f"\n[llm_engine] --- Chat Turn Start | Farm: '{farm_id}' | Lang: '{norm_lang}' ---"
    )

    user_prompts = [
        m.get("content", "").strip() for m in messages if m.get("role") == "user"
    ]
    last_user_query = user_prompts[-1] if user_prompts else ""

    # Step 1: Translate Hausa input to English for knowledge retrieval if needed
    effective_messages = []
    if norm_lang == "hausa":
        print("[llm_engine] Translating user input from Hausa to English...")
        for m in messages:
            if m.get("role") == "user":
                effective_messages.append(
                    {
                        "role": "user",
                        "content": translate_ha_to_en(m.get("content", "")),
                    }
                )
            else:
                effective_messages.append(m)
        current_query_en = (
            effective_messages[-1]["content"] if effective_messages else last_user_query
        )
    else:
        effective_messages = list(messages)
        current_query_en = last_user_query

    # Step 2: Instant Grounded Fast Intent Handler strictly for greetings
    fast_ans = handle_fast_intent(current_query_en, language=norm_lang)
    if fast_ans:
        print(
            f"[llm_engine] Fast greeting handler answered in {time.time() - turn_start:.2f}s"
        )
        return fast_ans

    llm = get_llm()
    if llm is None:
        return "[Fallback] Model not loaded."

    farm_summary = get_system_context_summary(farm_id)

    # Step 3: PASS 1 - JSON ROUTING (No logit bias applied here so it CAN output JSON)
    routing_system = get_routing_system_prompt(farm_id)

    # Build dialogue context from recent messages if multi-turn
    is_follow_up = len(current_query_en.split()) <= 6 or any(
        w in current_query_en.lower().split()
        for w in [
            "it",
            "its",
            "they",
            "them",
            "this",
            "these",
            "issue",
            "next",
            "why",
            "ok",
            "how",
        ]
    )
    if len(messages) > 1 and is_follow_up:
        recent_user_topics = [
            m["content"].strip()
            for m in messages[-4:-1]
            if m.get("role") == "user" and m.get("content")
        ]
        if recent_user_topics:
            router_query_content = f"Recent Topics: {'; '.join(recent_user_topics[-2:])}\nFarmer: '{current_query_en}'"
        else:
            router_query_content = f"Farmer: '{current_query_en}'"
    else:
        router_query_content = f"Farmer: '{current_query_en}'"

    routing_messages = [
        routing_system,
        {"role": "user", "content": router_query_content},
    ]
    grammar = get_llama_grammar()

    print("[llm_engine] Running Pass 1 (Router)...")
    response_pass1 = llm.create_chat_completion(
        messages=routing_messages,
        max_tokens=128,
        temperature=0.0,
        grammar=grammar,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    text_pass1 = response_pass1["choices"][0]["message"]["content"].strip()
    is_tool_call, tool_calls = parse_tool_calls(text_pass1)

    # Step 4: Execute Tools
    tool_results = []
    rag_context = ""

    if is_tool_call:
        for call in tool_calls:
            fn_name = call["function_name"]
            fn_args = call.get("arguments", {})
            print(f"[llm_engine] Executing tool '{fn_name}' with args {fn_args}")
            res = execute_tool(fn_name, fn_args, farm_id=farm_id)
            tool_results.append({"tool": fn_name, "result": res})

            if (
                fn_name == "query_knowledge_base"
                and isinstance(res, dict)
                and "context_prompt" in res
            ):
                rag_context = res["context_prompt"]

    if not is_tool_call and len(current_query_en) > 5:
        print("[llm_engine] Pass 1 yielded no tools, falling back to RAG safety net...")
        rag_hits = search_knowledge_base(current_query_en, top_k=2)
        relevant_hits = filter_relevant_rag_hits(rag_hits)

        if relevant_hits:
            rag_context = "\n---\n".join(
                [
                    f"[{h.get('filename')}]: {h.get('text')[:250]}"
                    for h in relevant_hits
                    if len(h.get("text", "")) > 50
                ]
            )
        else:
            print(
                "[llm_engine] No RAG hits cleared the relevance floor; will not synthesize from noise."
            )

    # Inject semantically matching farm memories into RAG context
    try:
        import farm_memory

        matching_mems = farm_memory.search_farm_memories(
            farm_id=farm_id, query=current_query_en, top_k=3
        )
        if matching_mems:
            mem_block = farm_memory.format_memories_for_rag(matching_mems)
            rag_context = f"{mem_block}\n\n{rag_context}" if rag_context else mem_block
            print(
                f"[llm_engine] Injected {len(matching_mems)} semantic farm memories into RAG context."
            )
    except Exception as e:
        print(f"[llm_engine] Semantic memory retrieval notice: {e}")

    # Step 5: SYNTHESIS / RESULT DISPATCH
    if rag_context:
        print("[llm_engine] Running Pass 3 (RAG Synthesis)...")
        raw_output = generate_stateless_answer(
            llm,
            rag_context,
            current_query_en,
            norm_lang,
            farm_summary,
            messages=messages,
        )
    elif tool_results:
        print("[llm_engine] Running Pass 3 (Database Synthesis)...")
        db_context = format_database_tool_context(tool_results)
        raw_output = generate_stateless_answer(
            llm,
            db_context,
            current_query_en,
            norm_lang,
            farm_summary,
            messages=messages,
        )
    else:
        print("[llm_engine] Running Direct Synthesis...")
        raw_output = generate_stateless_answer(
            llm,
            "",
            current_query_en,
            norm_lang,
            farm_summary,
            messages=messages,
        )

    # Step 6: Post-process & Language formatting
    if norm_lang == "hausa":
        print("[llm_engine] Translating English response to Hausa...")
        ha_translated = translate_en_to_ha(raw_output)
        religious_artifacts = [
            "littafi mai tsarki",
            "ãdalci",
            "sikẽlin",
            "la'ĩmi",
            "al'ummai",
            "karin magana",
        ]
        if any(art in ha_translated.lower() for art in religious_artifacts):
            print(
                "[llm_engine] Detected MarianMT religious artifact in translation. Falling back to English."
            )
            final_output = raw_output
        else:
            final_output = ha_translated
    else:  # English or Pidgin
        final_output = raw_output

    total_time = time.time() - turn_start
    print(f"[llm_engine] TOTAL TURN TIME: {total_time:.2f}s\n")
    return final_output


def chat_completion_stream(
    messages: list[dict[str, str]],
    farm_id: str = "default_farm",
    thread_id: str | None = None,
    language: str = "english",
):
    """Streaming generator yielding text tokens for unified multi-turn chat completion."""
    turn_start = time.time()
    norm_lang = normalize_language(language)
    print(
        f"\n[llm_engine] --- Chat Turn Stream Start | Farm: '{farm_id}' | Lang: '{norm_lang}' ---"
    )

    user_prompts = [
        m.get("content", "").strip() for m in messages if m.get("role") == "user"
    ]
    last_user_query = user_prompts[-1] if user_prompts else ""

    # Step 1: Translate Hausa input to English for knowledge retrieval if needed
    effective_messages = []
    if norm_lang == "hausa":
        print("[llm_engine] Translating user input from Hausa to English...")
        for m in messages:
            if m.get("role") == "user":
                effective_messages.append(
                    {
                        "role": "user",
                        "content": translate_ha_to_en(m.get("content", "")),
                    }
                )
            else:
                effective_messages.append(m)
        current_query_en = (
            effective_messages[-1]["content"] if effective_messages else last_user_query
        )
    else:
        effective_messages = list(messages)
        current_query_en = last_user_query

    # Step 2: Instant Grounded Fast Intent Handler strictly for greetings
    fast_ans = handle_fast_intent(current_query_en, language=norm_lang)
    if fast_ans:
        print(
            f"[llm_engine] Fast greeting handler answered in {time.time() - turn_start:.2f}s"
        )
        yield fast_ans
        return

    llm = get_llm()
    if llm is None:
        yield "[Fallback] Model not loaded."
        return

    farm_summary = get_system_context_summary(farm_id)

    # Step 3: PASS 1 - JSON ROUTING
    routing_system = get_routing_system_prompt(farm_id)

    is_follow_up = len(current_query_en.split()) <= 6 or any(
        w in current_query_en.lower().split()
        for w in [
            "it",
            "its",
            "they",
            "them",
            "this",
            "these",
            "issue",
            "next",
            "why",
            "ok",
            "how",
        ]
    )
    if len(messages) > 1 and is_follow_up:
        recent_user_topics = [
            m["content"].strip()
            for m in messages[-4:-1]
            if m.get("role") == "user" and m.get("content")
        ]
        if recent_user_topics:
            router_query_content = f"Recent Topics: {'; '.join(recent_user_topics[-2:])}\nFarmer: '{current_query_en}'"
        else:
            router_query_content = f"Farmer: '{current_query_en}'"
    else:
        router_query_content = f"Farmer: '{current_query_en}'"

    routing_messages = [
        routing_system,
        {"role": "user", "content": router_query_content},
    ]
    grammar = get_llama_grammar()

    print("[llm_engine] Running Pass 1 (Router)...")
    response_pass1 = llm.create_chat_completion(
        messages=routing_messages,
        max_tokens=128,
        temperature=0.0,
        grammar=grammar,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    text_pass1 = response_pass1["choices"][0]["message"]["content"].strip()
    is_tool_call, tool_calls = parse_tool_calls(text_pass1)

    # Step 4: Execute Tools
    tool_results = []
    rag_context = ""

    if is_tool_call:
        for call in tool_calls:
            fn_name = call["function_name"]
            fn_args = call.get("arguments", {})
            print(f"[llm_engine] Executing tool '{fn_name}' with args {fn_args}")
            res = execute_tool(fn_name, fn_args, farm_id=farm_id)
            tool_results.append({"tool": fn_name, "result": res})

            if (
                fn_name == "query_knowledge_base"
                and isinstance(res, dict)
                and "context_prompt" in res
            ):
                rag_context = res["context_prompt"]

    if not is_tool_call and len(current_query_en) > 5:
        print("[llm_engine] Pass 1 yielded no tools, falling back to RAG safety net...")
        rag_hits = search_knowledge_base(current_query_en, top_k=2)
        relevant_hits = filter_relevant_rag_hits(rag_hits)

        if relevant_hits:
            rag_context = "\n---\n".join(
                [
                    f"[{h.get('filename')}]: {h.get('text')[:250]}"
                    for h in relevant_hits
                    if len(h.get("text", "")) > 50
                ]
            )
        else:
            print(
                "[llm_engine] No RAG hits cleared the relevance floor; will not synthesize from noise."
            )

    # Inject semantically matching farm memories into RAG context
    try:
        import farm_memory

        matching_mems = farm_memory.search_farm_memories(
            farm_id=farm_id, query=current_query_en, top_k=3
        )
        if matching_mems:
            mem_block = farm_memory.format_memories_for_rag(matching_mems)
            rag_context = f"{mem_block}\n\n{rag_context}" if rag_context else mem_block
            print(
                f"[llm_engine] Injected {len(matching_mems)} semantic farm memories into RAG context."
            )
    except Exception as e:
        print(f"[llm_engine] Semantic memory retrieval notice: {e}")

    # Step 5: SYNTHESIS / RESULT DISPATCH
    target_context = (
        rag_context
        if rag_context
        else (format_database_tool_context(tool_results) if tool_results else "")
    )

    if norm_lang == "hausa":
        # For Hausa: generate English and translate
        raw_output = generate_stateless_answer(
            llm,
            target_context,
            current_query_en,
            norm_lang,
            farm_summary,
            messages=messages,
            stream=False,
        )
        ha_translated = translate_en_to_ha(raw_output)
        religious_artifacts = [
            "littafi mai tsarki",
            "ãdalci",
            "sikẽlin",
            "la'ĩmi",
            "al'ummai",
            "karin magana",
        ]
        if any(art in ha_translated.lower() for art in religious_artifacts):
            yield raw_output
        else:
            yield ha_translated
    else:
        # Stream tokens directly for English / Pidgin
        yield from generate_stateless_answer(
            llm,
            target_context,
            current_query_en,
            norm_lang,
            farm_summary,
            messages=messages,
            stream=True,
        )

    total_time = time.time() - turn_start
    print(f"[llm_engine] TOTAL TURN STREAM TIME: {total_time:.2f}s\n")
