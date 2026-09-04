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
        "--exp_id",
        type=int,
        default=1,
        help=(
            "Experiment number used for trained_cold/<date>/exp<ID>_cold_rna "
            "checkpoints and external_validation/<date>/exp<ID> results"
        )
    )
    parser.add_argument(
        "--date",
        type=str,
        default="undated",
        help=(
            "Date label used for both the pretrained checkpoint directory and "
            "external-validation result directory"
        )
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
        default=74,
        help="Dataset split seed (controls cross-validation and scaffold splits)"
    )

    # execution / evaluation
    parser.add_argument(
        "--setting",
        type=str,
        default="smi_emb_gen",
        choices=["cv", "loo_cv", "cold", "train_cold", "train_scaffold", "external", "smi_emb_gen"],
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
        default=2e-5,
        help="Training learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-6,
        help="Training weight decay for generalization control"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
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
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs used by training settings"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=None,
        help=(
            "Optional train_cold checkpoint output directory or external-evaluation "
            "checkpoint directory. By default saved_objects/trained_cold/<date>/"
            "exp<ID>_cold_rna is used."
        )
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
        "--save_predictions",
        type=str,
        default="False",
        choices=["True", "False"],
        help=(
            "Whether to save the best-epoch test predictions for every CV fold "
            "under results/table_results/<res_file_name>/"
        )
    )
    parser.add_argument(
        "--cold_type",
        type=str,
        default="cold_rm",
        choices=["cold_rna", "cold_mole", "cold_rm"],
        help="Cold-start type"
    )

    # SMILES embedding generation
    parser.add_argument(
        "--smiles_path",
        type=str,
        default="/home/gavrilev/Thesis/src/datasets/PDBBind2020/pdbbind_unique_smiles.pkl", # /home/gavrilev/Thesis/src/datasets/PDBBind2020/pdbbind_unique_smiles.pkl
        help="Path to pickled SMILES molecule set"
    )
    parser.add_argument(
        "--smiles_features_gen",
        nargs="+",
        default=["simson", "chemberta", "desc"],
        choices=["simson", "chemberta", "morgan_fp", "desc"],
        help="SMILES feature types to generate"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="smiles_emb_pdbbind", # smiles_emb_pdbbind for PDBBindNL2020 custom dataset, smiles_emb_enamine for case study
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


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        choices=[128, 256],
        help="Hidden dimension used inside BiDirectionalCrossFusion"
    )

    parser.add_argument(
        "--ffn_mult",
        type=int,
        default=4,
        choices=[2, 4, 6],
        help="Feed-forward expansion multiplier inside BiDirectionalCrossFusion"
    )

    parser.add_argument(
        "--bdcf_blocks",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of BiDirectionalCrossFusion blocks in the full model"
    )

    parser.add_argument(
        "--node_pool",
        type=str,
        choices=["attn", "mean"],
        default="mean",
        help="Pooling method for node embeddings: 'attn' or 'mean'"
    )

    parser.add_argument(
        "--token_pool",
        type=str,
        choices=["attn", "mean"],
        default="mean",
        help="Pooling method for token embeddings: 'attn' or 'mean'"
    )

    parser.add_argument(
        "--smiles_seq_setting",
        type=str,
        default="deep",
        choices=["proj", "shallow", "deep"],
        help="Architecture of the SMILES sequence encoder (ChemBERTa + SimSon): proj / shallow / deep"
    )

    parser.add_argument(
        "--smiles_expert_setting",
        type=str,
        default="proj",
        choices=["proj", "shallow", "deep"],
        help="Architecture of the SMILES expert encoder (Morgan FP + Mordred): proj / shallow / deep"
    )

    parser.add_argument(
        "--exclude_modalities",
        nargs="+",
        default=["morgan", "ernie"],
        choices=["simson", "chemberta", "mordred", "morgan", "rinalmo", "ernie", "grnade"],
        help="Modalities to exclude for sensitivity analysis"
    )
    parser.add_argument(
        "--freeze_grnade",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to freeze the gRNAde encoder"
    )

    parser.add_argument("--patience",
                        type=int,
                        default=20,
                        help="Patience for early stopping")

    parser.add_argument("--warm_up_epochs",
                        type=int,
                        default=10,
                        help="Number of warm-up epochs before early stopping starts")

    parser.add_argument("--delta",
                        type=float,
                        default=1e-6,
                        help="Minimum improvement required for early stopping")

    return parser.parse_args()
