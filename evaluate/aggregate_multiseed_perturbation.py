import argparse
import json
import os

import pandas as pd


METRICS = ["p", "s", "rmse", "r2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    seed_frames = []
    reference_permutations = None
    permutation_columns = [
        "fold", "perturbation", "source_rna_sequence",
        "perturbed_rna_sequence",
    ]

    for model_seed in args.model_seeds:
        seed_dir = os.path.join(args.input_dir, f"seed_{model_seed}")
        fold_summary_path = os.path.join(seed_dir, "fold_summary.csv")
        permutations_path = os.path.join(seed_dir, "structure_permutations.csv")
        if not os.path.isfile(fold_summary_path):
            raise FileNotFoundError(f"Missing seed summary: {fold_summary_path}")
        if not os.path.isfile(permutations_path):
            raise FileNotFoundError(
                f"Missing seed permutations: {permutations_path}")

        fold_df = pd.read_csv(fold_summary_path)
        expected_folds = list(range(1, 6))
        if sorted(fold_df["fold"].astype(int).tolist()) != expected_folds:
            raise ValueError(
                f"Seed {model_seed} does not contain folds {expected_folds}")
        if "model_seed" not in fold_df.columns or not (
                fold_df["model_seed"] == model_seed).all():
            raise ValueError(f"Incorrect model_seed metadata for seed {model_seed}")
        seed_frames.append(fold_df)

        permutations = pd.read_csv(permutations_path)[permutation_columns]
        permutations = permutations.sort_values(permutation_columns).reset_index(
            drop=True)
        if reference_permutations is None:
            reference_permutations = permutations
        elif not permutations.equals(reference_permutations):
            raise ValueError(
                f"Seed {model_seed} did not use the matched permutations")

    seed_fold_summary = pd.concat(seed_frames, ignore_index=True)
    seed_fold_summary = seed_fold_summary.sort_values(
        ["model_seed", "fold"]).reset_index(drop=True)
    seed_fold_summary.to_csv(
        os.path.join(args.input_dir, "seed_fold_summary.csv"), index=False)

    seed_rows = []
    for model_seed, fold_df in seed_fold_summary.groupby("model_seed", sort=True):
        row = {
            "model_seed": int(model_seed),
            "n_folds": len(fold_df),
            "n_perturbations": int(fold_df["n_perturbations"].iloc[0]),
        }
        for metric in METRICS:
            perturbed_value = float(
                fold_df[f"perturbation_{metric}_mean"].mean())
            row[f"perturbation_{metric}"] = perturbed_value
        seed_rows.append(row)

    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(
        os.path.join(args.input_dir, "seed_summary.csv"), index=False)

    paired_test_path = os.path.join(args.input_dir, "paired_t_tests.csv")
    if os.path.isfile(paired_test_path):
        os.remove(paired_test_path)

    config = {
        "model_seeds": args.model_seeds,
        "n_model_seeds": len(seed_summary),
        "n_folds_per_seed": 5,
        "n_perturbations": int(seed_summary["n_perturbations"].iloc[0]),
        "aggregation": (
            "perturbation metrics averaged over folds within each model seed"
        ),
        "matched_permutations_across_seeds": True,
    }
    with open(os.path.join(args.input_dir, "multiseed_run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Completed perturbation aggregation for model seeds {args.model_seeds}")


if __name__ == "__main__":
    main()
