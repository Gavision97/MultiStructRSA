"""Print the trainable parameter count for every MultiStructRSA setting."""

import models.MultiStructRSA as multistructrsa
from utils.parser import get_args


MODEL_SETTINGS = (
    "full",
    "freezed",
    "w/o seq",
    "w/o 3D struct",
    "w/o fusion",
    "full batched",
)


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    args = get_args()
    shared_hyperparameters = {
        "exclude_modalities": args.exclude_modalities,
        "freeze_grnade": args.freeze_grnade == "True",
        "smiles_seq_setting": args.smiles_seq_setting,
        "smiles_expert_setting": args.smiles_expert_setting,
        "dropout": args.dropout,
        "d_model": args.d_model,
        "ffn_mult": args.ffn_mult,
        "bdcf_blocks": args.bdcf_blocks,
        "node_pool": args.node_pool,
        "token_pool": args.token_pool,
    }

    for setting in MODEL_SETTINGS:
        # MultiStructRSABatched predates the configurable model factory and has
        # a parameterless constructor, so it uses its class-defined defaults.
        if setting == "full batched":
            # Its constructor references these legacy module-level dropout
            # names. They do not affect parameter counts, but assigning the
            # parser default lets the model be instantiated successfully.
            multistructrsa.BDCF_DROPOUT = args.dropout
            multistructrsa.DROPOUT = args.dropout
            multistructrsa.LST_DROPOUT = args.dropout
            model = multistructrsa.MultiStructRSABatched()
        else:
            model = multistructrsa.choose_model_setting(
                option=setting,
                **shared_hyperparameters,
            )

        print(f"{setting} -> {count_trainable_parameters(model):,}")


if __name__ == "__main__":
    main()
