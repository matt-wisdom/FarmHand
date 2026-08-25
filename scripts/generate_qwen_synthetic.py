import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
AWS_REGION = "eu-west-1"
MANTLE_BASE_URL = f"https://bedrock-mantle.{AWS_REGION}.api.aws/v1"
MANTLE_API_KEY = os.getenv("BEDROCK_MANTLE_API_KEY")
TEACHER_MODEL = "qwen.qwen3-235b-a22b-2507"

client = OpenAI(
    base_url=MANTLE_BASE_URL,
    api_key=MANTLE_API_KEY,
)

OUTPUT_FILE = "synthetic_farm_data_multiturn.jsonl"
PROGRESS_FILE = "generation_progress_multiturn.json"

TASKS = {
    "single_turn_tool": {
        "count": 1500,
        "description": "A direct 1-turn interaction where the user provides all necessary info, and the assistant immediately outputs the valid JSON tool call array.",
    },
    "multi_turn_clarification": {
        "count": 1500,
        "description": "A 2-turn interaction. Turn 1: User gives incomplete info. Assistant asks for missing parameters conversationally. Turn 2: User provides the missing data. Assistant outputs the JSON tool call array.",
    },
    "multi_turn_context": {
        "count": 500,
        "description": "A 2-to-3-turn interaction involving a general farming question answered conversationally (or via query_knowledge_base), followed by a related follow-up question or tool execution from the user.",
    },
    "guardrails": {
        "count": 500,
        "description": "A 1-turn interaction where the user asks about sports, politics, or non-farm topics, and the assistant politely refuses conversationally.",
    },
}

TOOL_SCHEMA = '[{"name": "write_expenditure", "description": "Log a financial transaction into the expenditures table.", "parameters": ["category", "amount", "description"]}, {"name": "write_health_log", "description": "Record an animal\'s medical or physical event into the health_logs table.", "parameters": ["animal_id", "event_type", "notes"]}, {"name": "get_sensor_data", "description": "Retrieve readings from the telemetry_data table.", "parameters": ["node_id", "sensor_type"]}, {"name": "get_animal_record", "description": "Retrieve an animal\'s demographic and current status from the animals table.", "parameters": ["id"]}, {"name": "trigger_vision_reid", "description": "Execute the MegaDetector and MegaDescriptor pipeline on a locally saved image to identify an animal.", "parameters": ["image_filepath"]}, {"name": "query_knowledge_base", "description": "Search the RAG vector database for farming advice, disease treatment, or general agricultural knowledge.", "parameters": ["search_query"]}]'


# ---------------------------------------------------------
# 2. Checkpoint & File Management
# ---------------------------------------------------------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load progress file ({e}). Starting fresh.")
    return {task: 0 for task in TASKS}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def sanitize_content(content):
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def append_conversations_and_update(task_type, conversations, progress):
    saved_count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for conversation in conversations:
            if not isinstance(conversation, list):
                continue

            chatml_row = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a localized farm management AI. You have access to the following tools: {TOOL_SCHEMA}",
                    }
                ]
            }

            valid_conversation = True
            for message in conversation:
                # Type guard against malformed non-dict items from LLM
                if not isinstance(message, dict):
                    valid_conversation = False
                    break

                role = message.get("role")
                content = sanitize_content(message.get("content", ""))

                if role not in ["user", "assistant"] or not content:
                    valid_conversation = False
                    break

                chatml_row["messages"].append({"role": role, "content": content})

            if valid_conversation and len(chatml_row["messages"]) > 1:
                f.write(json.dumps(chatml_row, ensure_ascii=False) + "\n")
                saved_count += 1

        f.flush()

    if saved_count > 0:
        progress[task_type] = progress.get(task_type, 0) + saved_count
        save_progress(progress)


def build_teacher_prompt(task_type, language):
    task_desc = TASKS[task_type]["description"]
    return f"""You are an expert AI data synthesizer building a dataset for an offline farm management system.
Generate 5 highly diverse, realistic synthetic conversations for this scenario: {task_desc}

Language Constraint: ALL user queries must be written in {language}. Assistant conversational responses must also be in {language} unless outputting a JSON tool call.
Setting: A rural/semi-urban farm in Nigeria managing crops, poultry, goats, and cattle.

ALLOWED TOOLS:
{TOOL_SCHEMA}

CRITICAL RULES:
1. Tool calls MUST be formatted as a raw JSON array. Example: [{{"function_name": "write_expenditure", "arguments": {{"category": "feed", "amount": 45000, "description": "2 bags goat feed"}}}}]
2. Do NOT invent tools or parameters outside the schema.
3. For multi-turn scenarios, simulate the back-and-forth logically.

Format output STRICTLY as a JSON array of 5 conversations. Each conversation is a JSON array of message objects.
Example Structure:
[
  [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
]
"""


# ---------------------------------------------------------
# 3. Execution Loop
# ---------------------------------------------------------
def generate_batch(task_type, language):
    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You output only raw JSON arrays. No markdown formatting, no code blocks, no conversational filler.",
                },
                {"role": "user", "content": build_teacher_prompt(task_type, language)},
            ],
            temperature=0.7,
            max_tokens=2500,
        )

        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()

        batch = json.loads(raw_content)

        # Unwrap if the LLM wrapped the entire batch in an outer array level
        if (
            isinstance(batch, list)
            and len(batch) == 1
            and isinstance(batch[0], list)
            and len(batch[0]) > 0
            and isinstance(batch[0][0], list)
        ):
            batch = batch[0]

        if isinstance(batch, list):
            return batch
    except Exception as e:
        print(f"Error generating batch for {task_type} ({language}): {e}")
    return []


def main():
    print(f"Starting Multi-Turn Generation (Model: {TEACHER_MODEL})...")
    progress = load_progress()

    languages = ["Nigerian Pidgin", "Hausa", "English"]
    lang_weights = [0.4, 0.4, 0.2]

    for task, config in TASKS.items():
        total_needed = config["count"]
        already_done = progress.get(task, 0)
        remaining_needed = total_needed - already_done

        if remaining_needed <= 0:
            print(
                f"\nTask '{task}' complete ({already_done}/{total_needed} rows). Skipping."
            )
            continue

        print(
            f"\nTask '{task}': {already_done}/{total_needed} completed. Generating {remaining_needed} remaining..."
        )

        batches_needed = (remaining_needed + 4) // 5

        for i in range(batches_needed):
            if i < batches_needed * lang_weights[0]:
                lang = languages[0]
            elif i < batches_needed * (lang_weights[0] + lang_weights[1]):
                lang = languages[1]
            else:
                lang = languages[2]

            batch = generate_batch(task, lang)
            if batch:
                append_conversations_and_update(task, batch, progress)
                print(
                    f"Progress for '{task}': {progress[task]}/{total_needed} conversations saved."
                )


if __name__ == "__main__":
    main()
