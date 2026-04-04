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

    # ---- seeds ----
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Training seed (controls model initialization, dataloader shuffle, etc.)"
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=2,
        help="Dataset split seed (controls cross-validation fold creation)"
    )

    # ---- evaluation settings ----
    parser.add_argument(
        "--model_setting",
        type=str,
        default="full",
        choices=["full", "freezed", "w/o seq", "w/o 3D struct", "w/o fusion", "full batched"],
        help=(
            "Model configuration to use. Options:\n"
            "  'full'            -> full model (3D + sequence + fusion)\n"
            "  'freezed'         -> same as full but with frozen gRNAde encoder\n"
            "  'w/o seq'         -> without sequence embeddings (ERNIE/RiNALMo)\n"
            "  'w/o 3D struct'   -> without gRNAde structural encoder\n"
            "  'w/o fusion'      -> without cross-attention fusion (simple concatenation)\n"
            "  'full batched'    -> full model (3D + seq + fusion); for batch_size > 1"
        )
    )
    parser.add_argument(
        "--is_seq_emb",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether its seuqence based embedding with shape (, F) or nucleotide-based shape (L, F) where L is sequence length and F is feature dimension"
    )   
    parser.add_argument(
        "--setting",
        type=str,
        default="cv",
        choices=["cv", "loo_cv", "cold", "smi_emb_gen"],
        help="execution setting (e.g., smiles feature generation, stratified k-fold CV evaluation, etc.)"
    )
    parser.add_argument(
        "--smiles_path",
        type=str,
        default="/home/gavrilev/Thesis/src/saved_objects/rsim_unique_smiles.pkl",
        help="Path to pickled SMILES molecule set"
    )
    parser.add_argument(
        "--smiles_features_gen",
        nargs="+",
        default=["simson", "chemberta", "morgan_fp", "desc"],
        choices=["simson", "chemberta", "morgan_fp", "desc"],
        help="SMILES feature types to generate"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="smiles_embeddings",
        help="generated SMILES embeddings save name; will be saved to -> embeddings/smiles/'save_name'.pkl"
    )
    parser.add_argument(
        "--cv_method",
        type=str,
        default="all",
        choices=["all", "per_fam"],
        help="Cross-validation split: full dataset or per RNA family"
    )
    
    parser.add_argument(
        "--ev_setting",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help=("# 1 - like in DeepRSMA, early stopping on the test set (test-leakage, overoptimistic) "
             "# 2 - early stopping on the validation loss & test best mode "
             "# 3 - early stopping on the validation loss & then retrain new model with chosen N epochs using train+val data"
             )
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )

    # ---- MolRNACDataset settings ----
    parser.add_argument(
        "--train_setting",
        type=int,
        default=6,
        help="gRNAde graph featurizer setting for training"
    )
    parser.add_argument(
        "--val_setting",
        type=int,
        default=4,
        help="gRNAde graph featurizer setting for validation/testing"
    )

    parser.add_argument(
        "--rna_aug",
        type=str,
        default="True",
        choices=["True", "False"],
        help="whether to use RNA data-augmentation while training"
    )

    parser.add_argument(
        "--mol_aug",
        type=str,
        default="False",
        choices=["True", "False"],
        help="whether to use SMILES data-augmentation while training"
    )


    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.9,
        help="Power-law smoothing exponent (alpha) for frequency-based negative sampling"
    )

    parser.add_argument(
        "--negative_factor",
        type=float,
        default=0.5,
        help="Number of synthetic negatives per positive sample (can be fractional)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="training learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
        help="training weight decay for generalization control"
    )
    parser.add_argument(
        "--is_plot",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to plot train vs. validation loss curve"
    )   
    # ---- cold-start settings ----
    parser.add_argument(
        "--cold_type",
        type=str,
        default="cold_rna",
        choices=["cold_rna", "cold_mol", "cold_rm"],
        help="Cold-start type"
    )

    # ---- logging & outputs ----
    parser.add_argument(
        "--plot_name",
        type=str,
        default="loss_curve",
        help="Name of saved loss curve plot (results/plots/<name>.png)"
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
        default="cpu",
        choices=["cuda", "cpu"],
        help="Computation device"
    )

    return parser.parse_args()