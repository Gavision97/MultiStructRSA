import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Training & evaluation configuration"
    )

    # runtime
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device"
    )

    # experiment metadata
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp",
        help="Experiment name (used for result tracking)"
    )
    parser.add_argument(
        "--log_msg",
        type=str,
        default="log message",
        help="Description message to include in the log file"
    )
    parser.add_argument(
        "--res_file_name",
        type=str,
        default="res",
        help="Name of the results DataFrame / output file"
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        default="eval_method_date_expnum",
        help="Logger file name"
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="loss_curve",
        help="Name of saved loss curve plot (results/plots/<name>.png)"
    )

    # seeds
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

    # execution / evaluation
    parser.add_argument(
        "--setting",
        type=str,
        default="cv",
        choices=["cv", "loo_cv", "cold", "smi_emb_gen"],
        help="Execution setting (e.g. SMILES feature generation, stratified k-fold CV evaluation, etc.)"
    )
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
        help="Whether input RNA embeddings are sequence-level (F,) or nucleotide-level (L, F), where L is sequence length and F is feature dimension"
    )
    parser.add_argument(
        "--ev_setting",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help=(
            "# 1 - like in DeepRSMA, early stopping on the test set (test leakage, overoptimistic) "
            "# 2 - early stopping on the validation loss & test best model "
            "# 3 - early stopping on the validation loss & then retrain new model with chosen N epochs using train+val data"
        )
    )

    # dataset / augmentation
    parser.add_argument(
        "--train_setting",
        type=int,
        default=4,
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
        default="False",
        choices=["True", "False"],
        help="Whether to use RNA data augmentation while training"
    )
    parser.add_argument(
        "--mol_aug",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to use SMILES data augmentation while training"
    )

    # training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Training learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
        help="Training weight decay for generalization control"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=1e-6,
        help="dropout value for regularization (same in both MLPs and BiDirectionalCrossFusion)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Train/val batch size (w/o aug=1, w > 1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader"
    )

    # cross-validation / splits
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
    parser.add_argument(
        "--cold_type",
        type=str,
        default="cold_rna",
        choices=["cold_rna", "cold_mol", "cold_rm"],
        help="Cold-start type"
    )

    # SMILES embedding generation
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
        help="Generated SMILES embeddings save name; will be saved to embeddings/smiles/<save_name>.pkl"
    )

    # visualization
    parser.add_argument(
        "--is_plot",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to plot train vs. validation loss curve"
    )

    return parser.parse_args()