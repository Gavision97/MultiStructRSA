import pandas as pd
import numpy as np



def generate_synthetic_negatives_for_train_fold(
    train_df,
    rna_col="Target_RNA_sequence",
    ligand_col="SMILES",
    pkd_col="pKd",
    threshold=4.0,
    alpha=0.75,
    negatives_per_positive=1.0,
    random_state=42,
    max_tries_per_negative=200,
):
    """
    Generate synthetic negative RNA–ligand interaction samples for a single
    training fold using a PPIMI-style frequency-based negative sampling strategy.

    Overview
    --------
    This procedure generates additional negative RNA–ligand pairs in order to
    alleviate data scarcity while preserving the statistical characteristics of
    the training fold. The method follows the PPIMI negative sampling logic,
    where the ligand is kept fixed and a new target (RNA) is sampled according
    to a smoothed frequency-based distribution.

    The algorithm operates strictly within the current training fold to avoid
    data leakage from validation or test folds.

    Methodology
    -----------
    1. The current training fold is used as the only source of data.
    2. Positive interactions are defined as rows with pKd >= threshold.
    3. RNA occurrence frequencies are computed from the training fold.
    4. Power-law smoothing is applied to the RNA frequencies using exponent
       `alpha`, producing a smoothed sampling distribution:
            P(RNA) ∝ frequency(RNA) ^ alpha
    5. Synthetic negatives are generated iteratively:
       a. Randomly select a positive RNA–ligand pair as the anchor.
       b. Keep the ligand (SMILES) fixed.
       c. Sample a new RNA from the smoothed RNA distribution.
       d. Reject the sampled RNA if:
            - it is identical to the anchor RNA, or
            - the resulting (RNA, ligand) pair already exists in the
              current training fold.
    6. A pseudo-negative pKd range is estimated from real weak-binding samples
       (pKd < threshold) in the training fold:
            Q1 = first quartile
            Q3 = third quartile
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            effective_lower_bound = max(min_observed_low_pkd, lower_bound)
    7. A pseudo-pKd value is sampled uniformly from:
            effective_lower_bound <= pKd < threshold
       This avoids extreme outliers while maintaining realistic weak-binding
       values.
    8. The generated synthetic negatives are appended to the original training
       fold and marked accordingly.

    Key Properties
    --------------
    - Uses only training-fold statistics (no data leakage)
    - Preserves ligand identity while sampling new RNA targets
    - Applies smoothed frequency-based sampling (power-law)
    - Prevents duplicate or known positive RNA–ligand pairs
    - Generates realistic pseudo-negative affinity values
    - Supports fractional negative sampling ratios

    Returns
    -------
    augmented_train_df : pd.DataFrame
        The original training fold concatenated with the generated synthetic
        negative samples.

    synthetic_neg_df : pd.DataFrame
        DataFrame containing only the generated synthetic negative samples.
    """
    # Initialize reproducible random generator
    rng = np.random.default_rng(random_state)

    # Copy input safely
    train_df = train_df.copy().reset_index(drop=True)

    # Validate required columns
    required_cols = [rna_col, ligand_col, pkd_col]
    missing_cols = [c for c in required_cols if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Store all existing (RNA, ligand) pairs in the current training fold
    existing_pairs = set(zip(train_df[rna_col], train_df[ligand_col]))

    # Select positive rows only
    pos_df = train_df[train_df[pkd_col] >= threshold].copy().reset_index(drop=True)
    if len(pos_df) == 0:
        raise ValueError(
            f"No positive rows found with {pkd_col} >= {threshold} in this training fold."
        )

    # Collect real low-pKd values from the current training fold
    raw_low_pkd = (
        train_df.loc[train_df[pkd_col] < threshold, pkd_col]
        .dropna()
        .astype(float)
    )
    if len(raw_low_pkd) == 0:
        raise ValueError(
            f"No real rows found with {pkd_col} < {threshold} in this training fold."
        )

    # Compute IQR-based lower bound for pseudo-negative pKd sampling
    q1 = raw_low_pkd.quantile(0.25)
    q3 = raw_low_pkd.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr

    # Prevent the lower bound from going below the observed minimum
    min_observed_low_pkd = raw_low_pkd.min()
    effective_lower_bound = max(min_observed_low_pkd, lower_bound)

    # Validate sampling range
    if effective_lower_bound >= threshold:
        raise ValueError(
            f"Invalid pseudo-pKd interval: [{effective_lower_bound}, {threshold})."
        )

    # Compute RNA frequencies from the current training fold
    rna_counts = train_df[rna_col].value_counts()

    # Apply power-law smoothing
    smoothed_counts = rna_counts.astype(float) ** alpha

    # Normalize to probabilities
    rna_probs = smoothed_counts / smoothed_counts.sum()

    # Prepare RNA candidates and probabilities
    rna_candidates = rna_probs.index.to_numpy()
    rna_candidate_probs = rna_probs.to_numpy()

    # Optional RNA metadata lookup to keep RNA metadata consistent
    rna_meta_cols = [c for c in ["Target_RNA_ID", "Target_RNA_name"] if c in train_df.columns]
    rna_lookup_cols = [rna_col] + rna_meta_cols
    rna_lookup = (
        train_df[rna_lookup_cols]
        .drop_duplicates(subset=[rna_col])
        .set_index(rna_col)
    )

    # Decide how many negatives to generate
    target_n_negatives = int(round(len(pos_df) * negatives_per_positive))

    # Handle zero requested negatives
    if target_n_negatives == 0:
        train_df["is_synthetic_negative"] = 0
        train_df["binary_label"] = (train_df[pkd_col] >= threshold).astype(int)
        synthetic_neg_df = pd.DataFrame(
            columns=train_df.columns.tolist() + [
                "is_synthetic_negative",
                "binary_label",
                "source_anchor_rna",
                "source_anchor_ligand",
            ]
        )
        return train_df, synthetic_neg_df

    # Storage for generated negatives
    synthetic_rows = []

    # Global attempt counter to avoid infinite loops
    tries = 0
    max_total_tries = target_n_negatives * max_tries_per_negative

    # Generate negatives until target is reached or tries are exhausted
    while len(synthetic_rows) < target_n_negatives and tries < max_total_tries:
        tries += 1

        # Choose one positive row as the anchor
        pos_row = pos_df.iloc[rng.integers(len(pos_df))]

        # Keep ligand fixed, sample a new RNA
        current_ligand = pos_row[ligand_col]
        original_rna = pos_row[rna_col]

        # Sample RNA from smoothed frequency distribution
        sampled_rna = rng.choice(rna_candidates, p=rna_candidate_probs)

        # Reject identical RNA
        if sampled_rna == original_rna:
            continue

        # Reject existing pair
        if (sampled_rna, current_ligand) in existing_pairs:
            continue

        # Start from anchor row and convert into synthetic negative
        syn_row = pos_row.copy()

        # Replace RNA identity
        syn_row[rna_col] = sampled_rna

        # Refresh RNA metadata if available
        if sampled_rna in rna_lookup.index:
            for meta_col in rna_meta_cols:
                syn_row[meta_col] = rna_lookup.loc[sampled_rna, meta_col]

        # Sample pseudo-pKd from continuous low-affinity interval
        sampled_pkd = rng.uniform(effective_lower_bound, threshold)
        sampled_pkd = min(sampled_pkd, np.nextafter(threshold, -np.inf))
        syn_row[pkd_col] = float(sampled_pkd)

        # Mark row
        syn_row["is_synthetic_negative"] = 1
        syn_row["binary_label"] = 0

        # Keep anchor info for sanity checks
        syn_row["source_anchor_rna"] = original_rna
        syn_row["source_anchor_ligand"] = current_ligand

        # Save and prevent duplicates
        synthetic_rows.append(syn_row)
        existing_pairs.add((sampled_rna, current_ligand))

    # Convert generated rows to dataframe
    synthetic_neg_df = pd.DataFrame(synthetic_rows)

    # Mark original training rows
    train_df["is_synthetic_negative"] = 0
    train_df["binary_label"] = (train_df[pkd_col] >= threshold).astype(int)

    # Concatenate original and synthetic rows
    augmented_train_df = pd.concat([train_df, synthetic_neg_df], ignore_index=True)

    return augmented_train_df, synthetic_neg_df

#####################################################################################33

def validate_synthetic_negatives(
    original_train_df,
    synthetic_neg_df,
    rna_col="Target_RNA_sequence",
    ligand_col="SMILES",
    pkd_col="pKd",
    threshold=4.0,
):
    """
    Validate the generated synthetic negatives.

    Checks
    ------
    1. All synthetic pKd values are strictly below threshold
    2. No synthetic (RNA, ligand) pair already existed in the original training fold
    3. No duplicate synthetic (RNA, ligand) pairs exist
    4. Synthetic marker column is correct
    5. Binary label column is correct
    6. Ligand was preserved relative to the anchor
    7. RNA was changed relative to the anchor
    """

    results = {}

    # Original training pairs
    original_pairs = set(zip(original_train_df[rna_col], original_train_df[ligand_col]))

    # Synthetic pairs
    synthetic_pairs = list(zip(synthetic_neg_df[rna_col], synthetic_neg_df[ligand_col]))
    synthetic_pair_set = set(synthetic_pairs)

    # Check pKd validity
    invalid_pkd_mask = synthetic_neg_df[pkd_col] >= threshold
    results["all_pkd_below_threshold"] = bool((~invalid_pkd_mask).all())
    results["n_invalid_pkd"] = int(invalid_pkd_mask.sum())

    # Check overlap with original training pairs
    overlapping_pairs = [pair for pair in synthetic_pair_set if pair in original_pairs]
    results["no_overlap_with_original_train"] = len(overlapping_pairs) == 0
    results["n_overlapping_pairs"] = len(overlapping_pairs)

    # Check duplicates inside synthetic negatives
    results["no_duplicate_synthetic_pairs"] = len(synthetic_pairs) == len(synthetic_pair_set)
    results["n_duplicate_synthetic_pairs"] = len(synthetic_pairs) - len(synthetic_pair_set)

    # Check synthetic marker
    if "is_synthetic_negative" in synthetic_neg_df.columns:
        bad_marker_mask = synthetic_neg_df["is_synthetic_negative"] != 1
        results["synthetic_marker_correct"] = bool((~bad_marker_mask).all())
        results["n_bad_synthetic_marker"] = int(bad_marker_mask.sum())
    else:
        results["synthetic_marker_correct"] = None
        results["n_bad_synthetic_marker"] = None

    # Check binary label
    if "binary_label" in synthetic_neg_df.columns:
        bad_binary_mask = synthetic_neg_df["binary_label"] != 0
        results["binary_label_correct"] = bool((~bad_binary_mask).all())
        results["n_bad_binary_label"] = int(bad_binary_mask.sum())
    else:
        results["binary_label_correct"] = None
        results["n_bad_binary_label"] = None

    # Check anchor ligand preservation
    if "source_anchor_ligand" in synthetic_neg_df.columns:
        same_ligand_mask = synthetic_neg_df[ligand_col] == synthetic_neg_df["source_anchor_ligand"]
        results["anchor_ligand_preserved"] = bool(same_ligand_mask.all())
        results["n_bad_anchor_ligand"] = int((~same_ligand_mask).sum())
    else:
        results["anchor_ligand_preserved"] = None
        results["n_bad_anchor_ligand"] = None

    # Check RNA was changed
    if "source_anchor_rna" in synthetic_neg_df.columns:
        changed_rna_mask = synthetic_neg_df[rna_col] != synthetic_neg_df["source_anchor_rna"]
        results["rna_was_changed"] = bool(changed_rna_mask.all())
        results["n_bad_same_rna"] = int((~changed_rna_mask).sum())
    else:
        results["rna_was_changed"] = None
        results["n_bad_same_rna"] = None

    # Summary counts
    results["n_original_train_rows"] = len(original_train_df)
    results["n_synthetic_rows"] = len(synthetic_neg_df)
    results["n_unique_synthetic_pairs"] = len(synthetic_pair_set)

    return results


######################################################################################33
def gen_neg_smoo_cold_fold(cold_folds=None, rna_col="Target_RNA_sequence", ligand_col="SMILES",
                           threshold=4.0, smoothing_factor=0.75, negative_factor=1):
    fold_outputs = []
    all_fold_checks = []
    neg_smoo_folds = {}
    for fold_idx, (curr_train_df, curr_test_df) in enumerate(cold_folds, start=1):
        print(f"\n===== Fold {fold_idx} =====")
    
        # Generate synthetic negatives only from the current training fold
        aug_train_df, synthetic_neg_df = generate_synthetic_negatives_for_train_fold(
            train_df=curr_train_df,
            rna_col=rna_col,
            ligand_col=ligand_col,
            pkd_col="pKd",
            threshold=threshold,
            alpha=smoothing_factor,
            negatives_per_positive=negative_factor, 
            random_state=42 + fold_idx,
            max_tries_per_negative=200,
        )
        neg_smoo_folds[f'fold{fold_idx}'] = (aug_train_df, curr_test_df)
        # Run sanity checks
        check_res = validate_synthetic_negatives(
            original_train_df=curr_train_df,
            synthetic_neg_df=synthetic_neg_df,
            rna_col=rna_col,
            ligand_col=ligand_col,
            pkd_col="pKd",
            threshold=threshold,
        )
    
        # Print compact summary
        print(f"Train rows:                 {len(curr_train_df)}")
        print(f"Test rows:                  {len(curr_test_df)}")
        print(f"Synthetic negatives:        {len(synthetic_neg_df)}")
        print(f"Augmented train rows:       {len(aug_train_df)}")
        print(f"All synthetic pKd < 4.0:    {check_res['all_pkd_below_threshold']}")
        print(f"No overlap with train:      {check_res['no_overlap_with_original_train']}")
        print(f"No duplicate synth pairs:   {check_res['no_duplicate_synthetic_pairs']}")
        print(f"Ligand preserved:           {check_res['anchor_ligand_preserved']}")
        print(f"RNA changed:                {check_res['rna_was_changed']}")
    
        # Hard sanity assertions
        assert check_res["all_pkd_below_threshold"], f"Fold {fold_idx}: found synthetic pKd >= threshold"
        assert check_res["no_overlap_with_original_train"], f"Fold {fold_idx}: synthetic pair overlaps original train"
        assert check_res["no_duplicate_synthetic_pairs"], f"Fold {fold_idx}: duplicate synthetic pairs found"
        assert check_res["anchor_ligand_preserved"], f"Fold {fold_idx}: ligand was not preserved"
        assert check_res["rna_was_changed"], f"Fold {fold_idx}: RNA was not changed"
    
        # Optional extra check: synthetic pairs should also not overlap with current test fold
        test_pairs = set(zip(curr_test_df[rna_col], curr_test_df[ligand_col]))
        synthetic_pairs = set(zip(synthetic_neg_df[rna_col], synthetic_neg_df[ligand_col]))
        overlap_with_test = synthetic_pairs & test_pairs
    
        print(f"No overlap with test pairs: {len(overlap_with_test) == 0}")
    
        # Save outputs
        fold_outputs.append({
            "fold": fold_idx,
            "train_df": curr_train_df,
            "test_df": curr_test_df,
            "aug_train_df": aug_train_df,
            "synthetic_neg_df": synthetic_neg_df,
        })
    
        all_fold_checks.append({
            "fold": fold_idx,
            "n_overlap_with_test_pairs": len(overlap_with_test),
            **check_res
        })
    
    return neg_smoo_folds, fold_outputs, pd.DataFrame(all_fold_checks)