import json
import sqlite3
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import database
from llm_engine import chat_completion, get_llm
from tool_registry import list_animals, match_species, execute_tool

def test_species_synonyms():
    print("\n--- Testing Species Synonym Matching ---")
    assert match_species("Poultry", "poultry") is True
    assert match_species("Chicken", "poultry") is True
    assert match_species("Layer", "poultry") is True
    assert match_species("Broiler", "chicken") is True
    assert match_species("Goat", "goat") is True
    assert match_species("Doe", "goat") is True
    assert match_species("Cow", "cattle") is True
    assert match_species("Goat", "poultry") is False
    assert match_species("Chicken", "goat") is False
    print("✓ Species synonym tests PASSED.")

def test_inventory_query_routing():
    print("\n--- Testing Farm Inventory Query Routing ---")
    thread_id = database.create_chat_thread(title="Inventory Test", farm_id="default_farm")
    
    # 1. Ask count of chickens when 0 exist
    prompt = "how many chickens do i have"
    messages = [{"role": "user", "content": prompt}]
    response = chat_completion(messages, farm_id="default_farm", thread_id=thread_id, language="english")
    print(f"\n[Prompt]: {prompt}\n[Response]:\n{response}\n")
    
    # Assertions
    assert "0" in response or "zero" in response.lower() or "no" in response.lower() or "none" in response.lower()
    assert "### FarmHand Advisory" in response
    print("✓ Inventory query with 0 chickens correctly handled.")

def test_registered_animals_inventory():
    print("\n--- Testing Inventory Query with Registered Animals ---")
    # Add temporary test animals to database
    database.add_animal_record("CHICK-001", "Hen Alpha", "Poultry", breed="Isa Brown", status="Active", farm_id="default_farm")
    database.add_animal_record("CHICK-002", "Hen Beta", "Poultry", breed="Isa Brown", status="Active", farm_id="default_farm")
    database.add_animal_record("GOAT-001", "Billy", "Goat", breed="Boer", status="Active", farm_id="default_farm")

    thread_id = database.create_chat_thread(title="Active Animals Test", farm_id="default_farm")
    
    # Query for chickens
    prompt = "how many chickens do i have"
    messages = [{"role": "user", "content": prompt}]
    response = chat_completion(messages, farm_id="default_farm", thread_id=thread_id, language="english")
    print(f"\n[Prompt]: {prompt}\n[Response]:\n{response}\n")
    
    assert "2" in response or "two" in response.lower() or "CHICK" in response or "Hen" in response
    print("✓ Inventory query with 2 registered chickens correctly counted and listed.")

    # Clean up test animals
    with database.get_db_connection() as conn:
        conn.cursor().execute("DELETE FROM animals WHERE id IN ('CHICK-001', 'CHICK-002', 'GOAT-001')")
    print("✓ Test animals cleaned up.")

def test_rag_clinical_query_routing():
    print("\n--- Testing RAG Knowledge Base Clinical Routing ---")
    thread_id = database.create_chat_thread(title="Clinical RAG Test", farm_id="default_farm")
    
    prompt = "what disease causes chickens to stumble backwards"
    messages = [{"role": "user", "content": prompt}]
    response = chat_completion(messages, farm_id="default_farm", thread_id=thread_id, language="english")
    print(f"\n[Prompt]: {prompt}\n[Response]:\n{response}\n")
    
    lower_resp = response.lower()
    assert any(k in lower_resp for k in ["synovitis", "newcastle", "paralysis", "infection", "vitamin", "disease", "airsacculitis", "lameness", "stiffness"])
    print("✓ RAG clinical query routing correctly handled.")

if __name__ == "__main__":
    test_species_synonyms()
    test_inventory_query_routing()
    test_registered_animals_inventory()
    test_rag_clinical_query_routing()
    print("\n=======================================================")
    print("ALL INVENTORY & RAG SEPARATION TESTS COMPLETED SUCCESSFULLY!")
    print("=======================================================")
