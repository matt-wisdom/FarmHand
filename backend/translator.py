import re
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

MODELS_DIR = Path(__file__).resolve().parent / "models" / "translator"
HA_EN_DIR = MODELS_DIR / "opus-mt-ha-en"
EN_HA_DIR = MODELS_DIR / "opus-mt-en-ha"

_ha_en_tok: MarianTokenizer | None = None
_ha_en_model: MarianMTModel | None = None

_en_ha_tok: MarianTokenizer | None = None
_en_ha_model: MarianMTModel | None = None


def get_ha_en_translator():
    global _ha_en_tok, _ha_en_model
    if _ha_en_model is None or _ha_en_tok is None:
        model_name_or_path = (
            str(HA_EN_DIR) if HA_EN_DIR.exists() else "Helsinki-NLP/opus-mt-ha-en"
        )
        local_only = HA_EN_DIR.exists()
        print(
            f"[translator] Loading ha->en model from {model_name_or_path} (local_only={local_only})..."
        )
        _ha_en_tok = MarianTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_only
        )
        _ha_en_model = MarianMTModel.from_pretrained(
            model_name_or_path, local_files_only=local_only
        )
        _ha_en_model.eval()
    return _ha_en_tok, _ha_en_model


def get_en_ha_translator():
    global _en_ha_tok, _en_ha_model
    if _en_ha_model is None or _en_ha_tok is None:
        model_name_or_path = (
            str(EN_HA_DIR) if EN_HA_DIR.exists() else "Helsinki-NLP/opus-mt-en-ha"
        )
        local_only = EN_HA_DIR.exists()
        print(
            f"[translator] Loading en->ha model from {model_name_or_path} (local_only={local_only})..."
        )
        _en_ha_tok = MarianTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_only
        )
        _en_ha_model = MarianMTModel.from_pretrained(
            model_name_or_path, local_files_only=local_only
        )
        _en_ha_model.eval()
    return _en_ha_tok, _en_ha_model


HAUSA_AGRI_GLOSSARY = [
    (r"\b(sannu da aiki|sannu da yamma|sannu da rana|sannu da safe|sannu)\b", "hello"),
    (r"\byaya aiki\b", "how is work"),
    (r"\byaya kake\b", "how are you"),
    (r"\byaya kuke\b", "how are you"),
    (r"\b(kaji na|kajina)\b", "my chickens"),
    (r"\bkaji\b", "chickens"),
    (r"\b(kaza ta|kazata)\b", "my chicken"),
    (r"\bkaza\b", "chicken"),
    (r"\b(awaki na|awakina)\b", "my goats"),
    (r"\bawaki\b", "goats"),
    (r"\b(akuya ta|akuyata)\b", "my goat"),
    (r"\bakuya\b", "goat"),
    (r"\b(shanu na|shanuna)\b", "my cattle"),
    (r"\bshanu\b", "cattle"),
    (r"\b(saniya ta|saniyata)\b", "my cow"),
    (r"\bsaniya\b", "cow"),
    (r"\b(tumaki na|tumakina)\b", "my sheep"),
    (r"\btumaki\b", "sheep"),
    (r"\b(gonata|gona ta)\b", "my farm"),
    (r"\bgona\b", "farm"),
    (r"\bdakin kaji\b", "poultry coop"),
    (r"\b(dabbobi na|dabbobina)\b", "my livestock"),
    (r"\bdabbobi\b", "livestock"),
    (r"\b(tsuntsaye na|tsuntsayena)\b", "my poultry birds"),
    (r"\btsuntsaye\b", "poultry birds"),
    (r"\bcuta\b", "disease"),
    (r"\bciwo\b", "sickness"),
    (r"\btari\b", "coughing"),
    (r"\bzazzabi\b", "fever"),
    (r"\bmura\b", "flu nasal discharge"),
    (r"\bmagani\b", "medicine treatment"),
    (r"\bnawa\b", "how many"),
]


def translate_ha_to_en(text: str) -> str:
    """Translates Hausa text into English using Helsinki-NLP/opus-mt-ha-en."""
    import time

    t0 = time.time()
    if not text or not text.strip():
        return text

    lower = text.strip().lower()
    clean_words = set(re.findall(r"\b\w+\b", lower))

    # Conversational Hausa greeting detection (if no specific farm task/disease/inventory is mentioned)
    greeting_terms = {
        "sannu",
        "barka",
        "kwana",
        "ina",
        "kake",
        "kuke",
        "yaya",
        "aiki",
        "godiya",
        "nagode",
        "gode",
    }
    agri_action_terms = {
        "kaji",
        "kaza",
        "awaki",
        "akuya",
        "shanu",
        "saniya",
        "tumaki",
        "rago",
        "dabbobi",
        "tsuntsaye",
        "cuta",
        "ciwo",
        "tari",
        "mura",
        "zazzabi",
        "magani",
        "nawa",
        "kudi",
        "kudin",
        "kashe",
        "rigakafi",
        "allura",
    }

    # Common conversational agricultural queries & count queries in Hausa
    if "nawa" in lower:
        if any(w in lower for w in ["dabbobi", "dabba", "gonata", "gona", "ke nan"]):
            res = "how many animals do i have?"
            print(
                f"\n[translator:ha->en] Quick query translation in 0.00s: '{text}' -> '{res}'\n"
            )
            return res
        if any(w in lower for w in ["kaji", "kaza", "tsuntsaye"]):
            res = "how many chickens do i have?"
            print(
                f"\n[translator:ha->en] Quick query translation in 0.00s: '{text}' -> '{res}'\n"
            )
            return res
        if any(w in lower for w in ["awaki", "akuya"]):
            res = "how many goats do i have?"
            print(
                f"\n[translator:ha->en] Quick query translation in 0.00s: '{text}' -> '{res}'\n"
            )
            return res
        if any(w in lower for w in ["shanu", "saniya"]):
            res = "how many cattle do i have?"
            print(
                f"\n[translator:ha->en] Quick query translation in 0.00s: '{text}' -> '{res}'\n"
            )
            return res

    if clean_words.intersection(greeting_terms) and not clean_words.intersection(
        agri_action_terms
    ):
        if "aiki" in clean_words:
            res = "hello, how is work?"
        elif "gode" in clean_words or "nagode" in clean_words:
            res = "thank you very much"
        else:
            res = "hello, how are you?"
        print(
            f"\n[translator:ha->en] Conversational translation in 0.00s: '{text}' -> '{res}'\n"
        )
        return res

    try:
        tok, model = get_ha_en_translator()
        inputs = tok(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=512)
        translated = tok.decode(translated_tokens[0], skip_special_tokens=True)

        # Enrich translation with species keywords if raw text clearly references livestock/poultry
        lower_raw = text.lower()
        prefix = ""
        if any(
            w in lower_raw for w in ["kaji", "kaza", "tsuntsaye", "dakin kaji"]
        ) and not any(
            w in translated.lower()
            for w in ["chicken", "poultry", "bird", "hen", "coop"]
        ):
            prefix += "poultry chickens "
        if any(w in lower_raw for w in ["awaki", "akuya"]) and not any(
            w in translated.lower() for w in ["goat", "buck", "doe"]
        ):
            prefix += "goats "
        if any(w in lower_raw for w in ["shanu", "saniya"]) and not any(
            w in translated.lower() for w in ["cattle", "cow", "bull"]
        ):
            prefix += "cattle cows "
        if any(w in lower_raw for w in ["tumaki", "rago"]) and not any(
            w in translated.lower() for w in ["sheep", "ram"]
        ):
            prefix += "sheep "

        result = f"{prefix}{translated}".strip()
        dt = time.time() - t0
        print(f"\n[translator:ha->en] Translation finished in {dt:.2f}s:")
        print(f"  [HA Input]      : {text}")
        print(f"  [EN Translated] : {result}\n")
        return result
    except Exception as e:
        print(f"[translator:ha->en] Error translating ha->en: {e}")
        return text


def translate_en_to_ha(text: str) -> str:
    """Translates English text into Hausa using Helsinki-NLP/opus-mt-en-ha."""
    import time

    t0 = time.time()
    if not text or not text.strip():
        return text

    try:
        # Preserve markdown header prefix if present
        header = ""
        body = text
        if text.startswith("### "):
            parts = text.split("\n\n", 1)
            if len(parts) == 2:
                header = "### Jagoran FarmHand:\n\n"
                body = parts[1]
            elif len(parts) == 1:
                return text

        tok, model = get_en_ha_translator()
        lines = body.split("\n")
        translated_lines = []

        for line in lines:
            if not line.strip():
                translated_lines.append("")
                continue
            inputs = tok(
                line, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                translated_tokens = model.generate(**inputs, max_length=512)
            translated = tok.decode(translated_tokens[0], skip_special_tokens=True)
            translated_lines.append(translated)

        translated_body = "\n".join(translated_lines)
        result = header + translated_body if header else translated_body
        dt = time.time() - t0
        print(f"\n[translator:en->ha] Translation finished in {dt:.2f}s:")
        print(f"  [EN Input]      : {text[:150]}...")
        print(f"  [HA Translated] : {result[:150]}...\n")
        return result
    except Exception as e:
        print(f"[translator:en->ha] Error translating en->ha: {e}")
        return text
