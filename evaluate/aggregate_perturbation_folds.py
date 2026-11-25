import argparse
import json
import os

import numpy as np
import pandas as pd


METRICS = ["p", "s", "rmse", "r2"]
RESULT_FILES = [
    "repeat_metrics.csv",
    "fold_summary.csv",
    "sample_predictions.csv",
    "structure_permutations.csv",
]


def summarize(fold_df):
    summary = {
        "model_setting": fold_df["model_setting"].iloc[0],
        "n_folds": len(fold_df),
        "n_perturbations": int(fold_df["n_perturbations"].iloc[0]),
    }
    if "model_seed" in fold_df.columns:
        summary["model_seed"] = int(fold_df["model_seed"].iloc[0])
    for metric in METRICS:
        for result_type, column in [
            ("perturbation", f"perturbation_{metric}_mean"),
            ("mean_prediction", f"mean_prediction_{metric}"),
        ]:
            values = fold_df[column].to_numpy(dtype=float)
            summary[f"{result_type}_{metric}_mean_across_folds"] = float(
                np.mean(values))
            summary[f"{result_type}_{metric}_std_across_folds"] = float(
                np.std(values))
    return pd.DataFrame([summary])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    fold_dirs = [
        os.path.join(args.input_dir, f"fold_{fold}")
        for fold in range(1, args.n_folds + 1)
    ]
    missing = [
        os.path.join(fold_dir, filename)
        for fold_dir in fold_dirs
        for filename in RESULT_FILES
        if not os.path.isfile(os.path.join(fold_dir, filename))
    ]
    if missing:
        raise FileNotFoundError(f"Missing fold result files: {missing}")

    combined = {}
    for filename in RESULT_FILES:
        frame = pd.concat(
            [pd.read_csv(os.path.join(fold_dir, filename))
             for fold_dir in fold_dirs],
            ignore_index=True,
        )
        if "fold" in frame.columns:
            sort_columns = ["fold"]
            if "perturbation" in frame.columns:
                sort_columns.append("perturbation")
            if "source_index" in frame.columns:
                sort_columns.append("source_index")
            frame = frame.sort_values(sort_columns).reset_index(drop=True)
        frame.to_csv(os.path.join(args.input_dir, filename), index=False)
        combined[filename] = frame

    fold_summary = combined["fold_summary.csv"]
    observed_folds = sorted(fold_summary["fold"].astype(int).tolist())
    expected_folds = list(range(1, args.n_folds + 1))
    if observed_folds != expected_folds:
        raise ValueError(
            f"Expected fold summaries {expected_folds}, got {observed_folds}")
    if fold_summary["n_perturbations"].nunique() != 1:
        raise ValueError("Fold results use different perturbation counts")

    summarize(fold_summary).to_csv(
        os.path.join(args.input_dir, "summary.csv"), index=False)

    first_config_path = os.path.join(fold_dirs[0], "run_config.json")
    if os.path.isfile(first_config_path):
        with open(first_config_path) as handle:
            config = json.load(handle)
        config["folds"] = expected_folds
        config["output_dir"] = args.input_dir
        config["aggregated_from"] = fold_dirs
        with open(os.path.join(args.input_dir, "run_config.json"), "w") as handle:
            json.dump(config, handle, indent=2)

    print(f"Aggregated {args.n_folds} folds in {args.input_dir}")


if __name__ == "__main__":
    main()
