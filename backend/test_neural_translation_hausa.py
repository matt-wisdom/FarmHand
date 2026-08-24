import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from database import init_db  # noqa: E402
from llm_engine import chat_completion  # noqa: E402
from translator import translate_en_to_ha, translate_ha_to_en  # noqa: E402


def test_pure_translation():
    print("=" * 50)
    print("TEST 1: Pure Translation (Opus-MT Hausa <-> English)")
    print("=" * 50)

    hausa_text = "Kaji na suna fama da tari da zazzabi."
    en_trans = translate_ha_to_en(hausa_text)
    print(f"[HA -> EN]:\n  Original: {hausa_text}\n  Translated: {en_trans}")
    assert len(en_trans) > 0, "Hausa to English translation failed!"

    english_text = (
        "Isolate sick chickens immediately and ensure the pen is dry and clean."
    )
    ha_trans = translate_en_to_ha(english_text)
    print(f"[EN -> HA]:\n  Original: {english_text}\n  Translated: {ha_trans}")
    assert len(ha_trans) > 0, "English to Hausa translation failed!"
    print("Pure translation test passed!\n")


def test_conversational_pidgin_and_english():
    print("=" * 50)
    print("TEST 2: Dynamic Routing on Conversational Inputs (No manual regex)")
    print("=" * 50)

    # 1. Pidgin greeting "how fa"
    print("Testing 'how fa' in Pidgin...")
    resp_pidgin = chat_completion(
        [{"role": "user", "content": "how fa"}], language="pidgin"
    )
    print(f"[Pidgin 'how fa' response]:\n{resp_pidgin}\n")
    assert resp_pidgin and len(resp_pidgin) > 10

    # 2. English greeting "hello"
    print("Testing 'hello' in English...")
    resp_en = chat_completion(
        [{"role": "user", "content": "hello"}], language="english"
    )
    print(f"[English 'hello' response]:\n{resp_en}\n")
    assert resp_en and len(resp_en) > 10
    print("Conversational dynamic routing test passed!\n")


def test_hausa_end_to_end_conversational():
    print("=" * 50)
    print("TEST 3: Hausa Conversational Greeting (End-to-End)")
    print("=" * 50)

    ha_greeting = "Sannu, yaya aiki?"
    print(f"User (Hausa): {ha_greeting}")
    resp = chat_completion([{"role": "user", "content": ha_greeting}], language="hausa")
    print(f"FarmHand (Hausa response):\n{resp}\n")
    assert resp and len(resp) > 5, "Empty response for Hausa greeting!"
    print("Hausa greeting test passed!\n")


def test_hausa_inventory_query():
    print("=" * 50)
    print("TEST 4: Hausa Farm Inventory Query (End-to-End)")
    print("=" * 50)

    init_db()
    # Query chickens in Hausa
    ha_query = "Kaji nawa nake da su a gonata?"
    print(f"User (Hausa): {ha_query}")
    resp = chat_completion(
        [{"role": "user", "content": ha_query}],
        farm_id="default_farm",
        language="hausa",
    )
    print(f"FarmHand (Hausa response):\n{resp}\n")
    assert resp and len(resp) > 5, "Empty response for Hausa inventory query!"
    print("Hausa inventory query test passed!\n")


def test_hausa_clinical_query():
    print("=" * 50)
    print("TEST 5: Hausa Clinical Disease Query (End-to-End)")
    print("=" * 50)

    ha_query = "Kaji na suna yin tari da mura da zubar da ruwa daga hanci, wace cuta ce kuma meye maganin?"
    print(f"User (Hausa): {ha_query}")
    resp = chat_completion(
        [{"role": "user", "content": ha_query}],
        farm_id="default_farm",
        language="hausa",
    )
    print(f"FarmHand (Hausa response):\n{resp}\n")
    assert resp and len(resp) > 10, "Empty response for Hausa clinical query!"
    print("Hausa clinical query test passed!\n")


if __name__ == "__main__":
    test_pure_translation()
    test_conversational_pidgin_and_english()
    test_hausa_end_to_end_conversational()
    test_hausa_inventory_query()
    test_hausa_clinical_query()
    print("=" * 50)
    print("ALL NEURAL TRANSLATION & DYNAMIC ROUTING TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
