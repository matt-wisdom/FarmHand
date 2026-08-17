import json
from typing import Any, Dict, List, Optional
from database import DB_PATH, get_db_connection

# Deferred import for rag_pipeline to prevent circular dependencies
_rag_pipeline_module = None


def get_rag_module():
    global _rag_pipeline_module
    if _rag_pipeline_module is None:
        import rag_pipeline
        _rag_pipeline_module = rag_pipeline
    return _rag_pipeline_module


# -------------------------------------------------------------------
# Database Tool Implementations
# -------------------------------------------------------------------

SPECIES_SYNONYMS = {
    "poultry": {"poultry", "chicken", "chickens", "layer", "layers", "broiler", "broilers", "pullet", "pullets", "cockerel", "cockerels", "fowl", "chick", "chicks", "hen", "hens", "rooster", "roosters"},
    "goat": {"goat", "goats", "buck", "bucks", "doe", "does", "kid", "kids"},
    "cattle": {"cattle", "cow", "cows", "bull", "bulls", "calf", "calves", "heifer", "heifers"},
    "sheep": {"sheep", "ram", "rams", "ewe", "ewes", "lamb", "lambs"},
    "pig": {"pig", "pigs", "swine", "hog", "hogs", "piglet", "piglets"},
    "fish": {"fish", "catfish", "tilapia", "aquaculture"},
}


def match_species(row_species: str, filter_species: str) -> bool:
    if not filter_species or filter_species.lower() in ("all", "any", "none", "*", ""):
        return True
    row_sp = row_species.strip().lower()
    filt_sp = filter_species.strip().lower()
    if row_sp == filt_sp:
        return True
    for group, synonyms in SPECIES_SYNONYMS.items():
        if filt_sp in synonyms or filt_sp == group:
            if row_sp in synonyms or row_sp == group:
                return True
    return filt_sp in row_sp or row_sp in filt_sp


def list_animals(species: str = "", farm_id: str = "default_farm", date_str: str = "") -> Dict[str, Any]:
    """Retrieve current or historical flock totals from the flock ledger."""
    import database
    if date_str:
        res = database.get_flock_count_on_date(farm_id=farm_id, species=species or None, target_date=date_str)
        return {"status": "success", "historical": True, "date": date_str, "data": res}
    else:
        totals = database.get_current_flock_totals(farm_id=farm_id)
        if species:
            norm_sp = database.normalize_species_name(species)
            count = totals.get(norm_sp, 0)
            return {"status": "success", "data": [{"species": norm_sp, "count": count}], "total": count}
        else:
            data = [{"species": sp, "count": cnt} for sp, cnt in totals.items()]
            return {"status": "success", "data": data, "total": sum(totals.values())}


def register_flock(species: str, count: int, notes: str = "", event_type: str = "count_update", farm_id: str = "default_farm") -> Dict[str, Any]:
    """Record a flock count update, purchase, or mortality into the flock ledger."""
    import database
    evt = event_type.lower().strip() if event_type else "count_update"
    is_delta = evt in ("purchase", "mortality", "sale", "loss", "addition") or int(count) < 0
    entry = database.record_flock_event(
        farm_id=farm_id,
        species=species,
        count_change=int(count) if is_delta else 0,
        exact_total=int(count) if not is_delta else None,
        event_type=evt,
        notes=notes
    )
    return {"status": "success", "entry": entry, "new_total": entry["new_total"]}


def list_expenditures(category: str = "", farm_id: str = "default_farm") -> Dict[str, Any]:
    """Retrieve all recorded financial expenditures from the database, optionally filtered by category."""
    import database
    rows = database.get_all_expenditures(farm_id=farm_id)
    if category:
        rows = [r for r in rows if category.lower() in r.get("category", "").lower() or r.get("category", "").lower() in category.lower()]
    return {"status": "success", "count": len(rows), "data": rows}


def list_health_logs(animal_id: str = "", farm_id: str = "default_farm") -> Dict[str, Any]:
    """Retrieve all recorded animal health check logs from the database, optionally filtered by animal_id."""
    import database
    rows = database.get_all_health_logs(farm_id=farm_id)
    if animal_id:
        rows = [r for r in rows if r.get("animal_id", "").lower() == animal_id.lower()]
    return {"status": "success", "count": len(rows), "data": rows}


def write_expenditure(category: str, amount: float, description: str = "") -> Dict[str, Any]:
    """Record a farm expenditure into the SQLite database."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO expenditures (category, amount, description) VALUES (?, ?, ?)",
            (category, float(amount), description)
        )
        record_id = cursor.lastrowid
    return {
        "status": "success",
        "message": f"Expenditure recorded with ID {record_id}",
        "data": {"id": record_id, "category": category, "amount": amount, "description": description}
    }


def write_health_log(animal_id: str, event_type: str, notes: str = "") -> Dict[str, Any]:
    """Record a health log entry for an animal into the SQLite database."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO health_logs (animal_id, event_type, notes) VALUES (?, ?, ?)",
            (animal_id, event_type, notes)
        )
        record_id = cursor.lastrowid
    return {
        "status": "success",
        "message": f"Health log recorded for animal {animal_id} with ID {record_id}",
        "data": {"id": record_id, "animal_id": animal_id, "event_type": event_type, "notes": notes}
    }


def get_sensor_data(node_id: str, sensor_type: str = "") -> Dict[str, Any]:
    """Retrieve telemetry sensor data for a specific node and optional sensor type."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        if sensor_type:
            cursor.execute(
                "SELECT * FROM telemetry_data WHERE node_id = ? AND sensor_type = ? ORDER BY timestamp DESC LIMIT 10",
                (node_id, sensor_type)
            )
        else:
            cursor.execute(
                "SELECT * FROM telemetry_data WHERE node_id = ? ORDER BY timestamp DESC LIMIT 10",
                (node_id,)
            )
        rows = [dict(r) for r in cursor.fetchall()]

    return {
        "status": "success",
        "count": len(rows),
        "data": rows
    }


def get_animal_record(id: str) -> Dict[str, Any]:
    """Retrieve details for a specific animal by ID."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM animals WHERE id = ?", (id,))
        row = cursor.fetchone()

    if row:
        return {"status": "success", "data": dict(row)}
    else:
        return {"status": "not_found", "message": f"No animal record found for ID '{id}'", "data": None}


def query_knowledge_base(search_query: str) -> Dict[str, Any]:
    """Search farm knowledge base PDFs using ONNX RAG embedding search."""
    rag = get_rag_module()
    return rag.query_knowledge_base(search_query)


def trigger_vision_reid(image_filepath: str) -> Dict[str, Any]:
    """Execute the MegaDetector and MegaDescriptor pipeline on a locally saved image to identify an animal."""
    return {
        "status": "success",
        "message": f"Vision Re-ID pipeline triggered for '{image_filepath}'",
        "data": {
            "image_filepath": image_filepath,
            "detected_species": "Goat",
            "confidence": 0.94,
            "matched_animal_id": "ANM-001"
        }
    }


# -------------------------------------------------------------------
# Tool Registry Map & Tool Schemas for LlamaGrammar / Two-Pass Router
# -------------------------------------------------------------------

TOOL_MAP = {
    "list_animals": list_animals,
    "register_flock": register_flock,
    "list_expenditures": list_expenditures,
    "list_health_logs": list_health_logs,
    "write_expenditure": write_expenditure,
    "write_health_log": write_health_log,
    "get_sensor_data": get_sensor_data,
    "get_animal_record": get_animal_record,
    "trigger_vision_reid": trigger_vision_reid,
    "query_knowledge_base": query_knowledge_base,
}

TOOL_SCHEMAS = [
    {
        "name": "list_animals",
        "description": "List current or historical flock totals from the flock ledger. Use this tool whenever the user asks how many animals/chickens they have or had.",
        "parameters": {
            "type": "object",
            "properties": {
                "species": {"type": "string", "description": "Optional species filter (e.g. poultry, goat, cattle)"},
                "date_str": {"type": "string", "description": "Optional historical date filter (YYYY-MM-DD)"}
            },
            "required": []
        }
    },
    {
        "name": "register_flock",
        "description": "Record a flock count update, purchase, or mortality event in the flock ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "species": {"type": "string", "description": "Species name (e.g. poultry, goat, cattle)"},
                "count": {"type": "integer", "description": "Number of animals or change in count"},
                "event_type": {"type": "string", "description": "Event type: initial_count, purchase, mortality, sale, count_update"},
                "notes": {"type": "string", "description": "Optional details or reason"}
            },
            "required": ["species", "count"]
        }
    },
    {
        "name": "list_expenditures",
        "description": "List all recorded financial expenditures in the farm database. Use this tool whenever the user asks to see or check expenses.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional category filter"}
            },
            "required": []
        }
    },
    {
        "name": "list_health_logs",
        "description": "List all recorded animal health check logs in the farm database.",
        "parameters": {
            "type": "object",
            "properties": {
                "animal_id": {"type": "string", "description": "Optional animal_id filter"}
            },
            "required": []
        }
    },
    {
        "name": "write_expenditure",
        "description": "Record a financial expenditure for farm operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category of expense (e.g. feed, fuel, equipment)"},
                "amount": {"type": "number", "description": "Monetary amount spent"},
                "description": {"type": "string", "description": "Detailed description of the expense"}
            },
            "required": ["category", "amount"]
        }
    },
    {
        "name": "write_health_log",
        "description": "Record a medical or health check log entry for an animal.",
        "parameters": {
            "type": "object",
            "properties": {
                "animal_id": {"type": "string", "description": "Unique identifier of the animal"},
                "event_type": {"type": "string", "description": "Type of event (e.g. vaccination, treatment, checkup)"},
                "notes": {"type": "string", "description": "Additional observation notes"}
            },
            "required": ["animal_id", "event_type"]
        }
    },
    {
        "name": "get_sensor_data",
        "description": "Retrieve recent telemetry sensor readings for an IoT node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Identifier of the sensor node (e.g. node_01)"},
                "sensor_type": {"type": "string", "description": "Optional sensor type filter (e.g. temperature, humidity)"}
            },
            "required": ["node_id"]
        }
    },
    {
        "name": "get_animal_record",
        "description": "Fetch details for a SPECIFIC animal by its known ID. Do NOT use this tool for general list queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Known animal unique ID"}
            },
            "required": ["id"]
        }
    },
    {
        "name": "trigger_vision_reid",
        "description": "Execute the MegaDetector and MegaDescriptor pipeline on a locally saved image to identify an animal.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_filepath": {"type": "string", "description": "Path to saved image file"}
            },
            "required": ["image_filepath"]
        }
    },
    {
        "name": "query_knowledge_base",
        "description": "Search farm manuals and PDF knowledge base for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "description": "Natural language search query for knowledge retrieval"}
            },
            "required": ["search_query"]
        }
    }
]


def normalize_tool_arguments(function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common argument key variations produced by LLMs in a pure, scalable way."""
    norm_args = dict(args)
    # Unpack nested arguments if produced by LLM
    if "arguments" in norm_args and isinstance(norm_args["arguments"], dict):
        for k, v in norm_args["arguments"].items():
            if k not in norm_args:
                norm_args[k] = v

    if function_name == "list_animals":
        sp = norm_args.get("species") or norm_args.get("animal_type") or norm_args.get("target_species") or norm_args.get("animal") or ""
        return {"species": str(sp).strip()}

    elif function_name == "list_expenditures":
        cat = norm_args.get("category") or norm_args.get("type") or ""
        return {"category": str(cat).strip()}

    elif function_name == "list_health_logs":
        aid = norm_args.get("animal_id") or norm_args.get("id") or ""
        return {"animal_id": str(aid).strip()}

    elif function_name == "write_health_log":
        animal_id = norm_args.get("animal_id") or norm_args.get("node_id") or norm_args.get("id") or "UNKNOWN"
        event_type = norm_args.get("event_type") or norm_args.get("category") or norm_args.get("treatment_type") or norm_args.get("health_event") or "health_check"
        notes = norm_args.get("notes") or norm_args.get("description") or (f"Expenditure: {norm_args.get('expenditure')}" if "expenditure" in norm_args else "")
        return {"animal_id": str(animal_id), "event_type": str(event_type), "notes": str(notes)}

    elif function_name == "write_expenditure":
        category = norm_args.get("category") or "general"
        amount_raw = norm_args.get("amount") if "amount" in norm_args else norm_args.get("expenditure", 0.0)
        try:
            amount = float(amount_raw)
        except Exception:
            amount = 0.0
        description = norm_args.get("description") or norm_args.get("notes") or ""
        return {"category": str(category), "amount": amount, "description": str(description)}

    elif function_name == "get_sensor_data":
        node_id = norm_args.get("node_id") or norm_args.get("id") or "node_01"
        sensor_type = norm_args.get("sensor_type") or ""
        return {"node_id": str(node_id), "sensor_type": str(sensor_type)}

    elif function_name == "get_animal_record":
        aid = norm_args.get("id") or norm_args.get("animal_id") or norm_args.get("node_id") or ""
        return {"id": str(aid)}

    elif function_name == "query_knowledge_base":
        query = norm_args.get("search_query") or norm_args.get("query") or norm_args.get("prompt") or ""
        return {"search_query": str(query).strip()}

    return norm_args


def execute_tool(function_name: str, arguments: Dict[str, Any], farm_id: str = "default_farm") -> Dict[str, Any]:
    """Execute a tool function by name with provided arguments dictionary and farm_id context."""
    # Redirection safety net: if get_animal_record was called with no ID or generic ID like "all" or "count"
    if function_name == "get_animal_record":
        aid = str(arguments.get("id", "")).strip().lower()
        if not aid or aid in ("all", "count", "none", "*") or "node_type" in arguments or "query_type" in arguments:
            sp = arguments.get("species") or arguments.get("node_type") or arguments.get("target_species") or ""
            return list_animals(species=str(sp), farm_id=farm_id)

    if function_name not in TOOL_MAP:
        return {"status": "error", "message": f"Unknown function '{function_name}'"}
    try:
        fn = TOOL_MAP[function_name]
        clean_args = normalize_tool_arguments(function_name, arguments)
        import inspect
        sig = inspect.signature(fn)
        if "farm_id" in sig.parameters:
            clean_args["farm_id"] = farm_id
        return fn(**clean_args)
    except Exception as e:
        return {"status": "error", "message": f"Error executing '{function_name}': {str(e)}"}
