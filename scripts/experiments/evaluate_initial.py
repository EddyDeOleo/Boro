
import csv
from pathlib import Path

RESULTS_PATH = Path("evaluation/initial_results.csv")


def main():
    if not RESULTS_PATH.is_file():
        print(f"File not found: {RESULTS_PATH}")
        return

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("The file contains no results.")
        return

    total = len(rows)
    correct = 0
    false_successes = 0
    missed_successes = 0

    print("\n=== BORO: PRELIMINARY EVALUATION ===\n")

    for row in rows:
        truth = row["ground_truth"].strip().lower()
        prediction = row["prediction"].strip().lower()
        agrees = truth == prediction

        correct += int(agrees)

        if truth == "failure" and prediction == "success":
            false_successes += 1

        if truth == "success" and prediction == "failure":
            missed_successes += 1

        status = "CORRECT" if agrees else "DISCREPANCY"

        print(
            f'{row["episode_id"]}: '
            f'ground_truth={truth}, prediction={prediction} '
            f'-> {status}'
        )

    accuracy = correct / total * 100

    print("\n=== SUMMARY ===")
    print(f"Episodes evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"False successes: {false_successes}")
    print(
        "Successful episodes incorrectly classified as failures: "
        f"{missed_successes}"
    )


if __name__ == "__main__":
    main()
