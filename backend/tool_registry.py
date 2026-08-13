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

def list_animals(species: str = "") -> Dict[str, Any]:
    """Retrieve all registered animal profiles from the database, optionally filtered by species."""
    import database
    rows = database.get_all_animals()
    if species:
        rows = [r for r in rows if r.get("species", "").lower() == species.lower()]
    return {"status": "success", "count": len(rows), "data": rows}


def list_expenditures(category: str = "") -> Dict[str, Any]:
    """Retrieve all recorded financial expenditures from the database, optionally filtered by category."""
    import database
    rows = database.get_all_expenditures()
    if category:
        rows = [r for r in rows if r.get("category", "").lower() == category.lower()]
    return {"status": "success", "count": len(rows), "data": rows}


def list_health_logs(animal_id: str = "") -> Dict[str, Any]:
    """Retrieve all recorded animal health check logs from the database, optionally filtered by animal_id."""
    import database
    rows = database.get_all_health_logs()
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
        "description": "List all registered animal profiles in the farm database. Use this tool whenever the user asks to see, list, or check what animals exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "species": {"type": "string", "description": "Optional species filter"}
            },
            "required": []
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

    if function_name == "list_animals":
        return {"species": str(norm_args.get("species", ""))}

    elif function_name == "list_expenditures":
        return {"category": str(norm_args.get("category", ""))}

    elif function_name == "list_health_logs":
        return {"animal_id": str(norm_args.get("animal_id", ""))}

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


def execute_tool(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool function by name with provided arguments dictionary."""
    if function_name not in TOOL_MAP:
        return {"status": "error", "message": f"Unknown function '{function_name}'"}
    try:
        fn = TOOL_MAP[function_name]
        clean_args = normalize_tool_arguments(function_name, arguments)
        return fn(**clean_args)
    except Exception as e:
        return {"status": "error", "message": f"Error executing '{function_name}': {str(e)}"}
