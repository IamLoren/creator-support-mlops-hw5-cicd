import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "scenario_id",
    "text",
    "language",
    "domain",
    "source",
    "intent",
}

ALLOWED_LANGUAGES = {"uk", "en"}

ALLOWED_DOMAINS = {
    "education",
    "fitness",
    "beauty",
    "professional_services",
}

ALLOWED_SOURCES = {
    "manual",
    "synthetic",
    "production_anonymized",
}

ALLOWED_INTENTS = {
    "ACCESS_ACCOUNT",
    "TECHNICAL_ISSUE",
    "SCHEDULE_DEADLINE",
    "SERVICE_INFO",
    "CONTENT_USAGE_QUESTION",
    "CHANGE_CANCEL",
    "FEEDBACK_COMPLAINT",
    "HUMAN_SUPPORT",
    "OTHER",
}

MESSAGE_ID_PATTERN = re.compile(r"^msg_\d{6}$")
SCENARIO_ID_PATTERN = re.compile(r"^scenario_\d{4}$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the canonical customer-support dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the canonical annotated dataset.",
    )

    parser.add_argument(
        "--expected-records",
        type=int,
        default=None,
        help="Optional expected number of records.",
    )

    parser.add_argument(
        "--expected-scenarios",
        type=int,
        default=None,
        help="Optional expected number of semantic scenarios.",
    )

    return parser.parse_args()


def validate_record(record: dict):
    missing = REQUIRED_FIELDS - record.keys()

    if missing:
        raise ValueError(
            f"Record {record.get('id')} missing fields: {sorted(missing)}"
        )

    if not MESSAGE_ID_PATTERN.fullmatch(record["id"]):
        raise ValueError(
            f"Invalid message id format: {record['id']}"
        )

    if not SCENARIO_ID_PATTERN.fullmatch(record["scenario_id"]):
        raise ValueError(
            f"Invalid scenario id format: {record['scenario_id']}"
        )

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(
            f"Record {record['id']} has empty text."
        )

    if record["language"] not in ALLOWED_LANGUAGES:
        raise ValueError(
            f"Invalid language in {record['id']}: {record['language']}"
        )

    if record["domain"] not in ALLOWED_DOMAINS:
        raise ValueError(
            f"Invalid domain in {record['id']}: {record['domain']}"
        )

    if record["source"] not in ALLOWED_SOURCES:
        raise ValueError(
            f"Invalid source in {record['id']}: {record['source']}"
        )

    if record["intent"] not in ALLOWED_INTENTS:
        raise ValueError(
            f"Invalid intent in {record['id']}: {record['intent']}"
        )


def main():
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON list.")

    if args.expected_records is not None:
        if len(records) != args.expected_records:
            raise ValueError(
                f"Expected {args.expected_records} records, "
                f"found {len(records)}."
            )

    ids = set()
    normalized_texts = set()

    scenarios = defaultdict(list)

    for record in records:
        validate_record(record)

        record_id = record["id"]

        if record_id in ids:
            raise ValueError(
                f"Duplicate record id: {record_id}"
            )

        ids.add(record_id)

        normalized_text = " ".join(
            record["text"].lower().split()
        )

        text_key = (
            record["language"],
            normalized_text,
        )

        if text_key in normalized_texts:
            raise ValueError(
                f"Duplicate text found: {record_id}"
            )

        normalized_texts.add(text_key)

        scenarios[record["scenario_id"]].append(record)

    if args.expected_scenarios is not None:
        if len(scenarios) != args.expected_scenarios:
            raise ValueError(
                f"Expected {args.expected_scenarios} scenarios, "
                f"found {len(scenarios)}."
            )

    for scenario_id, items in scenarios.items():
        if len(items) != 2:
            raise ValueError(
                f"{scenario_id} must contain exactly 2 records, "
                f"found {len(items)}."
            )

        languages = {
            item["language"]
            for item in items
        }

        if languages != {"uk", "en"}:
            raise ValueError(
                f"{scenario_id} must contain one uk and one en record. "
                f"Found: {sorted(languages)}"
            )

        domains = {
            item["domain"]
            for item in items
        }

        if len(domains) != 1:
            raise ValueError(
                f"{scenario_id} contains inconsistent domains: "
                f"{sorted(domains)}"
            )

        intents = {
            item["intent"]
            for item in items
        }

        if len(intents) != 1:
            raise ValueError(
                f"{scenario_id} contains inconsistent annotations: "
                f"{sorted(intents)}"
            )

    intent_counts = Counter(
        record["intent"]
        for record in records
    )

    language_counts = Counter(
        record["language"]
        for record in records
    )

    domain_counts = Counter(
        record["domain"]
        for record in records
    )

    print("Dataset validation PASSED")
    print(f"Records:   {len(records)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Unique IDs: {len(ids)}")

    print("\nLanguages:")
    for key in sorted(language_counts):
        print(f"  {key}: {language_counts[key]}")

    print("\nDomains:")
    for key in sorted(domain_counts):
        print(f"  {key}: {domain_counts[key]}")

    print("\nIntents:")
    for key in sorted(intent_counts):
        print(f"  {key}: {intent_counts[key]}")


if __name__ == "__main__":
    main()