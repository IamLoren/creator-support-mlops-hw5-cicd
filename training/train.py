import argparse
import json
from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DEFAULT_DATASET_PATH = Path(
    "data/annotated/customer_support_intents.json"
)

DEFAULT_OUTPUT_DIR = Path("artifacts")

DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_STATE = 42

# Hyperparameters selected in HW2
DEFAULT_C = 10.0
DEFAULT_NGRAM_MAX = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the customer-support intent classifier "
            "and save CI/CD artifacts."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the annotated dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model artifacts are stored.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of semantic scenarios used for final testing.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for deterministic splitting.",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=DEFAULT_C,
        help="Logistic Regression C parameter.",
    )

    parser.add_argument(
        "--ngram-max",
        type=int,
        default=DEFAULT_NGRAM_MAX,
        choices=[1, 2],
        help="Maximum TF-IDF n-gram size.",
    )

    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run 'dvc pull' first."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Dataset root must be a JSON array."
        )

    required_fields = {
        "id",
        "scenario_id",
        "text",
        "intent",
    }

    for index, record in enumerate(records):
        missing_fields = (
            required_fields - record.keys()
        )

        if missing_fields:
            raise ValueError(
                f"Record {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

    return records


def build_scenario_table(
    records: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Build one row per semantic scenario.

    Each bilingual pair shares scenario_id.
    All records belonging to one scenario must
    have the same intent.
    """

    scenario_to_intents: dict[str, set[str]] = {}

    for record in records:
        scenario_id = record["scenario_id"]
        intent = record["intent"]

        scenario_to_intents.setdefault(
            scenario_id,
            set(),
        ).add(intent)

    scenario_ids = []
    scenario_intents = []

    for scenario_id, intents in (
        scenario_to_intents.items()
    ):
        if len(intents) != 1:
            raise ValueError(
                f"Scenario {scenario_id} has "
                f"multiple intents: {sorted(intents)}"
            )

        scenario_ids.append(
            scenario_id
        )

        scenario_intents.append(
            next(iter(intents))
        )

    return (
        scenario_ids,
        scenario_intents,
    )


def split_by_scenario(
    records: list[dict],
    test_size: float,
    random_state: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split at scenario level so Ukrainian and English
    versions of the same semantic scenario never leak
    between development and test subsets.
    """

    (
        scenario_ids,
        scenario_intents,
    ) = build_scenario_table(records)

    (
        development_scenarios,
        test_scenarios,
    ) = train_test_split(
        scenario_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=scenario_intents,
    )

    development_scenarios = set(
        development_scenarios
    )

    test_scenarios = set(
        test_scenarios
    )

    development_records = [
        record
        for record in records
        if record["scenario_id"]
        in development_scenarios
    ]

    test_records = [
        record
        for record in records
        if record["scenario_id"]
        in test_scenarios
    ]

    return (
        development_records,
        test_records,
    )


def build_model(
    c_value: float,
    ngram_max: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(
                        1,
                        ngram_max,
                    ),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def save_json(
    path: Path,
    data: dict,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=float,
        )


def main():
    args = parse_args()

    records = load_dataset(
        args.dataset
    )

    (
        development_records,
        test_records,
    ) = split_by_scenario(
        records=records,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    x_development = [
        record["text"]
        for record in development_records
    ]

    y_development = [
        record["intent"]
        for record in development_records
    ]

    x_test = [
        record["text"]
        for record in test_records
    ]

    y_test = [
        record["intent"]
        for record in test_records
    ]

    model = build_model(
        c_value=args.c,
        ngram_max=args.ngram_max,
    )

    print("Training model...")

    model.fit(
        x_development,
        y_development,
    )

    print(
        "Evaluating model on held-out test set..."
    )

    predictions = model.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    classification_report_data = (
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    )

    development_scenarios = {
        record["scenario_id"]
        for record in development_records
    }

    test_scenarios = {
        record["scenario_id"]
        for record in test_records
    }

    metrics = {
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }

    metadata = {
        "model_type": (
            "TF-IDF + Logistic Regression"
        ),
        "dataset_version": "dataset-v0.2",
        "dataset_records": len(
            records
        ),
        "development_records": len(
            development_records
        ),
        "test_records": len(
            test_records
        ),
        "development_scenarios": len(
            development_scenarios
        ),
        "test_scenarios": len(
            test_scenarios
        ),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "tfidf_ngram_range": [
            1,
            args.ngram_max,
        ],
        "logistic_regression_C": (
            args.c
        ),
        "max_iter": 1000,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        args.output_dir
        / "model.joblib"
    )

    metrics_path = (
        args.output_dir
        / "metrics.json"
    )

    metadata_path = (
        args.output_dir
        / "model_metadata.json"
    )

    report_path = (
        args.output_dir
        / "classification_report.json"
    )

    joblib.dump(
        model,
        model_path,
    )

    save_json(
        metrics_path,
        metrics,
    )

    save_json(
        metadata_path,
        metadata,
    )

    save_json(
        report_path,
        classification_report_data,
    )

    print()
    print("Training completed.")
    print()
    print(
        f"Total records: "
        f"{len(records)}"
    )
    print(
        f"Development records: "
        f"{len(development_records)}"
    )
    print(
        f"Test records: "
        f"{len(test_records)}"
    )
    print(
        f"Development scenarios: "
        f"{len(development_scenarios)}"
    )
    print(
        f"Test scenarios: "
        f"{len(test_scenarios)}"
    )

    print()
    print("Test metrics:")
    print(
        f"Accuracy:    "
        f"{accuracy:.4f}"
    )
    print(
        f"Macro F1:    "
        f"{macro_f1:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print()
    print("Artifacts:")
    print(
        f"Model: "
        f"{model_path}"
    )
    print(
        f"Metrics: "
        f"{metrics_path}"
    )
    print(
        f"Metadata: "
        f"{metadata_path}"
    )
    print(
        f"Classification report: "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()