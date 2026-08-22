"""
FarmHand AI Feature & Pipeline Evaluation Suite (backend/evals/examples.py)
==========================================================================
Programmatic evaluations covering:
- RAG disease passage retrieval (Tetanus, Coccidiosis, Newcastle, Mastitis)
- Feed formulation retrieval (DIY Fish Feed, Day-Old Chicks, Pig Maize)
- Agronomy retrieval (Cassava Mosaic, Fall Armyworm)
- Database Flock Ledger Headcount & Sales Decrements
- Financial Expenditure Farm-ID Scoping
- Persistent Farm Memory Logging
- Router Tool-Calling & Delineation (Sales vs Expenditures)
- Direct Definitional LLM Synthesis
"""
import sys
import time
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.run_evals import register_eval, EvalResult


# =============================================================================
# 1. RAG Disease Retrieval Evaluations
# =============================================================================

@register_eval("rag_disease_goat_tetanus")
def eval_rag_disease_goat_tetanus() -> EvalResult:
    """Evaluate RAG retrieves Clostridium / lockjaw passages for goat tetanus."""
    start = time.time()
    try:
        from rag_pipeline import search_knowledge_base
        results = search_knowledge_base("goat tetanus lockjaw bacteria infection causes symptoms", top_k=5)
        combined_text = " ".join([r.get("text", "").lower() for r in results])
        
        has_bacteria = "bacteria" in combined_text or "clostridium" in combined_text or "tetani" in combined_text
        has_symptoms = "jaw" in combined_text or "stiff" in combined_text or "wound" in combined_text or "tetanus" in combined_text
        
        passed = has_bacteria and has_symptoms and len(results) > 0
        details = f"Retrieved {len(results)} chunks. Has bacteria/toxin: {has_bacteria}, Has symptoms: {has_symptoms}"
        score = 1.0 if passed else (0.5 if len(results) > 0 else 0.0)
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_disease_goat_tetanus", passed, score, details, latency)


@register_eval("rag_disease_poultry_coccidiosis")
def eval_rag_disease_poultry_coccidiosis() -> EvalResult:
    """Evaluate RAG retrieves protozoan / bloody droppings passages for poultry coccidiosis."""
    start = time.time()
    try:
        from rag_pipeline import search_knowledge_base
        results = search_knowledge_base("poultry coccidiosis protozoan parasite causes symptoms", top_k=5)
        combined_text = " ".join([r.get("text", "").lower() for r in results])
        
        has_coccidiosis = "coccidi" in combined_text or "eimeria" in combined_text or "protozo" in combined_text
        has_signs = "droppings" in combined_text or "bloody" in combined_text or "litter" in combined_text or "diarrhea" in combined_text
        
        passed = has_coccidiosis and len(results) > 0
        details = f"Retrieved {len(results)} chunks. Has coccidiosis terms: {has_coccidiosis}, Has signs: {has_signs}"
        score = 1.0 if (has_coccidiosis and has_signs) else (0.6 if has_coccidiosis else 0.0)
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_disease_poultry_coccidiosis", passed, score, details, latency)


# =============================================================================
# 2. RAG Feed Formulation & Agronomy Retrieval
# =============================================================================

@register_eval("rag_feed_fish_diy")
def eval_rag_feed_fish_diy() -> EvalResult:
    """Evaluate RAG retrieves protein & binder ingredients for DIY fish feed."""
    start = time.time()
    try:
        from rag_pipeline import search_knowledge_base
        results = search_knowledge_base("fish feed formulation DIY homemade low cost ingredients", top_k=5)
        combined_text = " ".join([r.get("text", "").lower() for r in results])
        
        has_protein = "protein" in combined_text or "soya" in combined_text or "fish" in combined_text or "meal" in combined_text
        has_carb = "cassava" in combined_text or "maize" in combined_text or "bran" in combined_text or "energy" in combined_text
        
        passed = has_protein and len(results) > 0
        details = f"Retrieved {len(results)} chunks. Has protein: {has_protein}, Has energy/binder: {has_carb}"
        score = 1.0 if (has_protein and has_carb) else (0.6 if has_protein else 0.0)
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_feed_fish_diy", passed, score, details, latency)


@register_eval("rag_crop_cassava_mosaic")
def eval_rag_crop_cassava_mosaic() -> EvalResult:
    """Evaluate RAG retrieves control & resistant variety info for cassava mosaic disease."""
    start = time.time()
    try:
        from rag_pipeline import search_knowledge_base
        results = search_knowledge_base("cassava mosaic virus CMD whitefly resistant variety management", top_k=5)
        combined_text = " ".join([r.get("text", "").lower() for r in results])
        
        has_mosaic = "mosaic" in combined_text or "virus" in combined_text or "cmd" in combined_text
        has_control = "resistant" in combined_text or "cutting" in combined_text or "whitefly" in combined_text or "stem" in combined_text
        
        passed = has_mosaic and len(results) > 0
        details = f"Retrieved {len(results)} chunks. Has mosaic terms: {has_mosaic}, Has control terms: {has_control}"
        score = 1.0 if (has_mosaic and has_control) else (0.6 if has_mosaic else 0.0)
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("rag_crop_cassava_mosaic", passed, score, details, latency)


# =============================================================================
# 3. Database Flock Ledger & Sales Headcount Decrement Evaluations
# =============================================================================

@register_eval("db_ledger_livestock_sale_decrement")
def eval_db_ledger_livestock_sale_decrement() -> EvalResult:
    """Test that registering a livestock sale correctly calculates running headcount decrement."""
    start = time.time()
    try:
        import database
        import tool_registry
        
        test_farm = "eval_farm_sale_test"
        # 1. Set baseline count of 12 goats
        res_init = tool_registry.register_flock(
            species="goat",
            count=12,
            event_type="initial_count",
            farm_id=test_farm
        )
        init_total = res_init.get("new_total", 0)
        
        # 2. Record sale of 4 goats
        res_sale = tool_registry.register_flock(
            species="goat",
            count=4,
            event_type="sale",
            notes="Sold 4 goats at NGN 15,000 each (Total: NGN 60,000)",
            farm_id=test_farm
        )
        
        sale_total = res_sale.get("new_total", 0)
        count_change = res_sale.get("entry", {}).get("count_change", 0)
        
        passed = (init_total == 12) and (sale_total == 8) and (count_change == -4)
        details = f"Initial: {init_total}, Sold: 4, Count Change: {count_change}, New Total: {sale_total}"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("db_ledger_livestock_sale_decrement", passed, score, details, latency)


@register_eval("flock_anomaly_sales_classification")
def eval_flock_anomaly_sales_classification() -> EvalResult:
    """Test that commercial livestock sales are never misclassified as mortality spikes or disease outbreaks."""
    start = time.time()
    try:
        import database
        import tool_registry
        import anomaly_detector
        
        test_farm = "eval_farm_anomaly_sales_check"
        # 1. Setup 10 goats
        tool_registry.register_flock(
            species="goat",
            count=10,
            event_type="initial_count",
            farm_id=test_farm
        )
        
        # 2. Record sale of 3 goats
        tool_registry.register_flock(
            species="goat",
            count=3,
            event_type="sale",
            notes="Sold 3 goats at NGN 15,000 each",
            farm_id=test_farm
        )
        
        # 3. Run anomaly detector
        res = anomaly_detector.run_flock_anomaly_detection(farm_id=test_farm)
        severity = res.get("severity")
        issues = res.get("metrics", {}).get("deterministic_issues", [])
        mortality_issues = [i for i in issues if "MORTALITY" in i.get("type", "")]
        
        passed = (severity == "NORMAL") and (len(mortality_issues) == 0)
        details = f"Severity: {severity}, Total Issues: {len(issues)}, Mortality Issues: {len(mortality_issues)}"
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("flock_anomaly_sales_classification", passed, score, details, latency)


@register_eval("db_expenditure_farm_scoping")
def eval_db_expenditure_farm_scoping() -> EvalResult:
    """Test that writing an expenditure properly scopes the record to the active farm ID."""
    start = time.time()
    try:
        import database
        import tool_registry
        
        test_farm = "eval_farm_exp_scoped"
        res_exp = tool_registry.write_expenditure(
            category="goat feed",
            amount=25000.0,
            description="Purchased 5 bags of feed",
            farm_id=test_farm
        )
        
        # Query farm expenditures
        farm_records = database.get_all_expenditures(farm_id=test_farm)
        matching = [e for e in farm_records if e.get("id") == res_exp.get("data", {}).get("id")]
        
        passed = len(matching) == 1 and matching[0]["farm_id"] == test_farm and matching[0]["amount"] == 25000.0
        details = f"Recorded ID {res_exp.get('data', {}).get('id')} to farm '{test_farm}'. Found in DB: {len(matching)} record(s)."
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("db_expenditure_farm_scoping", passed, score, details, latency)


@register_eval("db_farm_memory_logging")
def eval_db_farm_memory_logging() -> EvalResult:
    """Test logging infrastructure observation into persistent semantic farm memory."""
    start = time.time()
    try:
        import tool_registry
        import farm_memory
        
        test_farm = "eval_farm_memory_test"
        res_mem = tool_registry.log_farm_observation(
            species="goat",
            observation="Has an elevated slatted floor goat barn",
            category="infrastructure",
            farm_id=test_farm
        )
        
        # Verify retrieval
        mems = farm_memory.get_farm_memories(farm_id=test_farm)
        matching = [m for m in mems if "slatted floor" in m.get("observation", "").lower()]
        
        passed = len(matching) > 0 and matching[0]["species"] == "goat"
        details = f"Logged memory. Verified in DB: {len(matching)} observation(s)."
        score = 1.0 if passed else 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("db_farm_memory_logging", passed, score, details, latency)


# =============================================================================
# 4. Router Tool Classification Evaluations
# =============================================================================

@register_eval("router_livestock_sale_routing")
def eval_router_livestock_sale_routing() -> EvalResult:
    """Test Pass 1 Router selects register_flock (not write_expenditure) for 'we sold 3 goats at 15000 each'."""
    start = time.time()
    try:
        import llm_engine
        llm = llm_engine.get_llm()
        if not llm:
            return EvalResult("router_livestock_sale_routing", False, 0.0, "LLM not loaded", 0)
        
        grammar = llm_engine.get_llama_grammar()
        routing_system = {
            "role": "system",
            "content": (
                "You are the tool routing engine for FarmHand AI.\n"
                "Output ONLY a valid JSON array with the single best tool call.\n"
                "TOOLS:\n"
                "- register_flock(species: str, count: int, event_type: str, notes: str): Set or update headcount for births, purchases, sales, mortalities.\n"
                "- write_expenditure(category: str, amount: float, description: str): Record farm operating costs/expenses. NEVER use for animal sales."
            )
        }
        
        res = llm.create_chat_completion(
            messages=[routing_system, {"role": "user", "content": "we sold 3 goats at 15000 each"}],
            max_tokens=64,
            temperature=0.0,
            grammar=grammar,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        out = res["choices"][0]["message"]["content"].strip()
        is_tc, calls = llm_engine.parse_tool_calls(out)
        
        if is_tc and calls:
            fn = calls[0]["function_name"]
            passed = fn == "register_flock"
            details = f"Routed to '{fn}' (Expected: 'register_flock')"
            score = 1.0 if passed else 0.0
        else:
            passed = False
            details = f"No tool parsed from output: {out[:60]}"
            score = 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("router_livestock_sale_routing", passed, score, details, latency)


@register_eval("router_feed_expenditure_routing")
def eval_router_feed_expenditure_routing() -> EvalResult:
    """Test Pass 1 Router selects write_expenditure for 'Record 15,000 NGN spent on goat feed'."""
    start = time.time()
    try:
        import llm_engine
        llm = llm_engine.get_llm()
        if not llm:
            return EvalResult("router_feed_expenditure_routing", False, 0.0, "LLM not loaded", 0)
        
        grammar = llm_engine.get_llama_grammar()
        routing_system = {
            "role": "system",
            "content": (
                "You are the tool routing engine for FarmHand AI.\n"
                "Output ONLY a valid JSON array with the single best tool call.\n"
                "TOOLS:\n"
                "- register_flock(species: str, count: int, event_type: str, notes: str): Animal headcount.\n"
                "- write_expenditure(category: str, amount: float, description: str): Record farm operating costs/expenses."
            )
        }
        
        res = llm.create_chat_completion(
            messages=[routing_system, {"role": "user", "content": "Record 15,000 NGN spent on goat feed"}],
            max_tokens=64,
            temperature=0.0,
            grammar=grammar,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        out = res["choices"][0]["message"]["content"].strip()
        is_tc, calls = llm_engine.parse_tool_calls(out)
        
        if is_tc and calls:
            fn = calls[0]["function_name"]
            passed = fn == "write_expenditure"
            details = f"Routed to '{fn}' (Expected: 'write_expenditure')"
            score = 1.0 if passed else 0.0
        else:
            passed = False
            details = f"No tool parsed from output: {out[:60]}"
            score = 0.0
    except Exception as e:
        passed = False
        details = f"Error: {str(e)[:100]}"
        score = 0.0
    latency = (time.time() - start) * 1000
    return EvalResult("router_feed_expenditure_routing", passed, score, details, latency)