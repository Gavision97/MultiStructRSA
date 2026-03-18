import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Training & evaluation configuration"
    )

    # ---- experiment metadata ----
    parser.add_argument(
        "--log_msg",
        type=str,
        default="log message",
        help="Description message to include in the log file"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp",
        help="Experiment name (used for result tracking)"
    )
    parser.add_argument(
        "--res_file_name",
        type=str,
        default="res",
        help="Name of the results DataFrame / output file"
    )

    # ---- evaluation settings ----
    parser.add_argument(
        "--model_setting",
        type=str,
        default="full",
        choices=["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion"],
        help=(
            "Model configuration to use. Options:\n"
            "  'full'            -> full model (3D + sequence + fusion)\n"
            "  'freezed'         -> same as full but with frozen gRNAde encoder\n"
            "  'w/o seq'         -> without sequence embeddings (ERNIE/RiNALMo)\n"
            "  'w/o 3D struct'   -> without gRNAde structural encoder\n"
            "  'w/o fusion'      -> without cross-attention fusion (simple concatenation)"
              )
    )   
    parser.add_argument(
        "--eval_method",
        type=str,
        default="cv",
        choices=["cv", "loo_cv", "cold_blind_cv"],
        help="Evaluation strategy"
    )
    parser.add_argument(
        "--cv_method",
        type=str,
        default="all",
        choices=["all", "per_fam"],
        help="Cross-validation split: full dataset or per RNA family"
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )

    # ---- cold-start settings ----
    parser.add_argument(
        "--cold_type",
        type=str,
        default="rna",
        choices=["rna", "smiles", "rna_and_smiles"],
        help="Cold-start type"
    )

    # ---- logging & outputs ----
    parser.add_argument(
        "--plot_name",
        type=str,
        default="loss_curve",
        help="Name of saved loss curve plot (results/plots/<name>.png); format -> 'date_plotName' (e.g., '1404_exp7')"
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        default="eval_method_date_expnum",
        help="Logger file name"
    )

    # ---- runtime ----
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device"
    )

    return parser.parse_args()