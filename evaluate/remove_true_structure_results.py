"""Remove legacy true-structure outputs from perturbation result directories."""

import argparse
import csv
import json
import os


def is_true_dependent_column(column):
    return (
        column == "true_structure_prediction"
        or column.startswith("true_structure_")
        or column.startswith("improvement_")
    )


def rewrite_csv(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return False

    keep_indices = [
        index for index, column in enumerate(rows[0])
        if not is_true_dependent_column(column)
    ]
    if len(keep_indices) == len(rows[0]):
        return False

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [[row[index] for index in keep_indices] for row in rows]
        )
    return True


def rewrite_multiseed_config(path):
    with open(path) as handle:
        old_config = json.load(handle)

    model_seeds = old_config.get("model_seeds", [])
    config = {
        "model_seeds": model_seeds,
        "n_model_seeds": len(model_seeds),
        "n_folds_per_seed": old_config.get("n_folds_per_seed", 5),
        "n_perturbations": old_config.get("n_perturbations"),
        "aggregation": (
            "perturbation metrics averaged over folds within each model seed"
        ),
        "matched_permutations_across_seeds": old_config.get(
            "matched_permutations_across_seeds", True
        ),
    }
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def rewrite_log(path):
    marker = " - INFO - Completed perturbation analysis\n"
    with open(path) as handle:
        content = handle.read()
    marker_index = content.find(marker)
    if marker_index == -1:
        return False

    marker_end = marker_index + len(marker)
    replacement = content[:marker_end]
    replacement += (
        "Perturbation-only summary retained in summary.csv; "
        "legacy comparison columns removed.\n"
    )
    if replacement == content:
        return False
    with open(path, "w") as handle:
        handle.write(replacement)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(args.input_dir)

    removed_files = []
    rewritten_csvs = []
    rewritten_logs = []

    paired_test_path = os.path.join(args.input_dir, "paired_t_tests.csv")
    if os.path.isfile(paired_test_path):
        os.remove(paired_test_path)
        removed_files.append(paired_test_path)

    for root, _, files in os.walk(args.input_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if filename.endswith(".csv") and rewrite_csv(path):
                rewritten_csvs.append(path)
            elif filename == "perturbation_analysis.log" and rewrite_log(path):
                rewritten_logs.append(path)

    multiseed_config_path = os.path.join(
        args.input_dir, "multiseed_run_config.json"
    )
    if os.path.isfile(multiseed_config_path):
        rewrite_multiseed_config(multiseed_config_path)

    print(f"Rewritten CSV files: {len(rewritten_csvs)}")
    print(f"Rewritten log files: {len(rewritten_logs)}")
    print(f"Removed fully comparison-dependent files: {len(removed_files)}")


if __name__ == "__main__":
    main()
