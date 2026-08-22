#!/usr/bin/env python3
"""
FarmHand Evaluation Creator Tool (evals/add_eval.py)
---------------------------------------------------
Easily add new benchmark evaluation cases to the eval dataset.

Usage:
  # Interactive mode:
  python evals/add_eval.py

  # CLI argument mode:
  python evals/add_eval.py \
    --id eval_vet_swine_fever \
    --category veterinary_diagnosis \
    --language english \
    --query "My pigs have high fever, red skin blotches, and are dying rapidly. What could be the cause?" \
    --must-contain "african swine fever,virus,biosecurity" \
    --description "African Swine Fever symptom diagnosis"
"""

import sys
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT_DIR / "evals" / "eval_dataset.json"

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def add_eval(
    query: str,
    category: str = "general",
    language: str = "english",
    eval_id: str | None = None,
    description: str = "",
    must_contain: list[str] | None = None,
    must_not_contain: list[str] | None = None,
    expected_tools: list[str] | None = None,
    prose_required: bool = False,
    no_forced_list: bool = False,
    farm_id: str = "farm_9eb3f441",
    dataset_path: Path = DEFAULT_DATASET
) -> dict:
    """
    Appends a new evaluation case to the specified dataset file.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        dataset = {"version": "1.0", "description": "FarmHand AI Benchmark Suite", "evals": []}
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    if not eval_id:
        count = len(dataset.get("evals", [])) + 1
        clean_cat = category.lower().replace(" ", "_")
        eval_id = f"eval_{clean_cat}_{count:03d}_{language[:2]}"

    # Verify uniqueness
    for existing in dataset.get("evals", []):
        if existing.get("id") == eval_id:
            raise ValueError(f"An evaluation with ID '{eval_id}' already exists in {dataset_path}")

    # Standard negative particles for English mode
    final_must_not = list(must_not_contain or [])
    if language == "english":
        default_pidgin_prohibited = ["dey", "wey", "wetin", "una", "dem", " di ", "well-well"]
        for p in default_pidgin_prohibited:
            if p not in final_must_not:
                final_must_not.append(p)

    new_eval = {
        "id": eval_id,
        "category": category,
        "language": language.lower(),
        "description": description or f"{category.replace('_', ' ').title()} query in {language}",
        "farm_id": farm_id,
        "messages": [
            {
                "role": "user",
                "content": query.strip()
            }
        ],
        "expected_tools": expected_tools or ["query_knowledge_base"],
        "must_contain": [k.strip() for k in (must_contain or []) if k.strip()],
        "must_not_contain": [k.strip() for k in final_must_not if k.strip()]
    }

    if prose_required:
        new_eval["prose_required"] = True
    if no_forced_list:
        new_eval["no_forced_list"] = True

    dataset["evals"].append(new_eval)

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    return new_eval


def interactive_wizard(dataset_path: Path):
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}                FarmHand Benchmark: Add New Evaluation                 {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

    query = input(f"{BOLD}1. Farmer's Question / Prompt:{RESET} ").strip()
    while not query:
        query = input(f"{RED}Question cannot be empty. Enter prompt:{RESET} ").strip()

    lang_choice = input(f"{BOLD}2. Language (1 = english [default], 2 = pidgin):{RESET} ").strip()
    language = "pidgin" if lang_choice == "2" else "english"

    print(f"\n{BOLD}Select Category:{RESET}")
    categories = [
        "veterinary_diagnosis",
        "feed_formulation",
        "agronomy_crops",
        "aquaculture",
        "tool_calling",
        "multi_turn_diagnosis",
        "general"
    ]
    for i, c in enumerate(categories, 1):
        print(f"  {i}) {c}")
    cat_input = input(f"Choice (1-{len(categories)}) [default: 1]: ").strip()
    try:
        category = categories[int(cat_input) - 1] if cat_input else categories[0]
    except (ValueError, IndexError):
        category = "general"

    eval_id = input(f"{BOLD}3. Unique Eval ID (leave blank to auto-generate):{RESET} ").strip() or None
    description = input(f"{BOLD}4. Description / Evaluation goal:{RESET} ").strip()

    must_contain_str = input(f"{BOLD}5. Keywords that MUST appear in the answer (comma-separated):{RESET} ").strip()
    must_contain = [t.strip() for t in must_contain_str.split(",") if t.strip()]

    must_not_str = input(f"{BOLD}6. Terms that MUST NOT appear (comma-separated, optional):{RESET} ").strip()
    must_not_contain = [t.strip() for t in must_not_str.split(",") if t.strip()]

    prose_choice = input(f"{BOLD}7. Require paragraph prose (no forced lists)? (y/N):{RESET} ").strip().lower()
    prose_required = prose_choice in ["y", "yes"]

    try:
        created = add_eval(
            query=query,
            category=category,
            language=language,
            eval_id=eval_id,
            description=description,
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            prose_required=prose_required,
            no_forced_list=prose_required,
            dataset_path=dataset_path
        )
        print(f"\n{GREEN}{BOLD}✓ Successfully added evaluation '{created['id']}' to {dataset_path}!{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Error adding evaluation: {e}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new evaluation to FarmHand dataset")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Path to eval dataset JSON")
    parser.add_argument("--query", type=str, default=None, help="Farmer input prompt")
    parser.add_argument("--id", type=str, default=None, help="Unique evaluation ID")
    parser.add_argument("--category", type=str, default="general", help="Evaluation category")
    parser.add_argument("--language", type=str, default="english", choices=["english", "pidgin"], help="Language mode")
    parser.add_argument("--description", type=str, default="", help="Description of test goal")
    parser.add_argument("--must-contain", type=str, default="", help="Comma-separated keywords required in answer")
    parser.add_argument("--must-not-contain", type=str, default="", help="Comma-separated forbidden words")
    parser.add_argument("--expected-tools", type=str, default="query_knowledge_base", help="Comma-separated tool names")
    parser.add_argument("--prose-required", action="store_true", help="Require natural paragraph prose (no lists)")
    parser.add_argument("--no-forced-list", action="store_true", help="Require no numbered list format")

    args = parser.parse_args()

    if args.query:
        must_c = [t.strip() for t in args.must_contain.split(",") if t.strip()]
        must_not = [t.strip() for t in args.must_not_contain.split(",") if t.strip()]
        exp_tools = [t.strip() for t in args.expected_tools.split(",") if t.strip()]
        res = add_eval(
            query=args.query,
            category=args.category,
            language=args.language,
            eval_id=args.id,
            description=args.description,
            must_contain=must_c,
            must_not_contain=must_not,
            expected_tools=exp_tools,
            prose_required=args.prose_required,
            no_forced_list=args.no_forced_list,
            dataset_path=Path(args.dataset)
        )
        print(f"{GREEN}✓ Evaluation '{res['id']}' added successfully!{RESET}")
    else:
        interactive_wizard(Path(args.dataset))
