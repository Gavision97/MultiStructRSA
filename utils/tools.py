import os
import random
import numpy as np
import pandas as pd
import torch


from rdkit import Chem
from sklearn.preprocessing import StandardScaler

import logging
logger = logging.getLogger(__name__)

COLD_PATH = 'datasets/blind_test/'

#############################################################################################
"""
SMILES randomization adapted from:

Bjerrum, E. J. (2017).
"SMILES enumeration as data augmentation for neural network modeling of molecules."
arXiv:1703.07076

Official implementation:
https://github.com/Ebjerrum/SMILES-enumeration
"""
def randomize_smiles(smiles, canonical=False, isomericSmiles=True):
    """Randomize SMILES atom ordering for data augmentation"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None

    atom_order = list(range(m.GetNumAtoms()))
    np.random.shuffle(atom_order)

    m = Chem.RenumberAtoms(m, atom_order)
    return Chem.MolToSmiles(
        m,
        canonical=canonical,
        isomericSmiles=isomericSmiles
    )

#############################################################################################
def enumerate_unique(smiles, n=10, max_tries=100):
    """
    Generate up to n unique randomized SMILES for the same molecule
    using atom-order randomization.
    """
    seen = {smiles}
    tries = 0

    while len(seen) < n and tries < max_tries:
        s = randomize_smiles(smiles)
        if s is not None:
            seen.add(s)
        tries += 1

    return list(seen)

#################################################################################################3
# same style as their code
def set_seed(seed_):
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)



#def seed_worker(worker_id):
#    worker_seed = torch.initial_seed() % 2**32
#    np.random.seed(worker_seed)
#    random.seed(worker_seed)
#################################################################################################

def load_cold_data(cold_type="cold_rna"):
    _ ,cold = cold_type.split('_')
    df1 = pd.read_csv(COLD_PATH + cold + '/' + cold_type +'1.csv', delimiter=',')
    df2 = pd.read_csv(COLD_PATH + cold + '/' + cold_type +'2.csv', delimiter=',')
    df3 = pd.read_csv(COLD_PATH + cold + '/' + cold_type +'3.csv', delimiter=',')
    df4 = pd.read_csv(COLD_PATH + cold + '/' + cold_type +'4.csv', delimiter=',')
    df5 = pd.read_csv(COLD_PATH + cold + '/' + cold_type +'5.csv', delimiter=',')
    cold_dfs = [df1, df2, df3, df4, df5]
    return cold_dfs

def get_cold_splits(curr_cold_df=None, cold_dfs=None, all_df=None, fold=0):
    """
    Generate train/test indices for a given cold-start fold.

    Args:
        curr_cold_df (pd.DataFrame): current fold dataframe used as test set.
        cold_dfs (list[pd.DataFrame]): list of all cold split dataframes.
        all_df (pd.DataFrame): full dataset containing Entry_ID column.
        fold (int): index of the current test fold.

    Returns:
        tuple[list[int], list[int]]: train indices, test indices.
    """
    if curr_cold_df is None or cold_dfs is None or all_df is None:
        raise ValueError("Either curr_cold_df/cold_dfs/all_df is None")

    test_id = curr_cold_df['Entry_ID'].tolist()
    test_id = all_df[all_df['Entry_ID'].isin(test_id)].index.tolist()

    train_id = []
    for j, other_df in enumerate(cold_dfs):
        if j != fold:
            train_id.extend(other_df['Entry_ID'].tolist())

    train_id = all_df[all_df['Entry_ID'].isin(train_id)].index.tolist()
    return train_id, test_id
#################################################################################################


# NOTE -> Scale after splitting to train/test (val) in order to avoid data leakage!
def scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard-scale all numeric feature columns (assumes first column is SMILES/ID),
    then return a DataFrame with the original first column + scaled features.
    """
    if df.shape[0] <= 2:
        return df

    smiles_col = df.iloc[:, 0].copy()
    X = df.iloc[:, 1:].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    scaled_df.insert(0, df.columns[0], smiles_col)

    return scaled_df   


def scale_nonzero(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numeric feature columns in a DataFrame using StandardScaler,
    excluding rows where all feature values are zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. The first column is assumed to be non-feature (e.g., ID, SMILES, etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled features (excluding all-zero rows from scaling).
    """
    if df.shape[0] <= 2:
        return df

    res_df = df.copy()
    feature_columns = res_df.columns[1:]  # assume first column is not a feature
    scaler = StandardScaler()

    # mask rows that are not all zeros
    non_zero_mask = ~(res_df[feature_columns] == 0).all(axis=1)

    # fit and transform only non-zero rows
    scaler.fit(res_df.loc[non_zero_mask, feature_columns])
    res_df.loc[non_zero_mask, feature_columns] = scaler.transform(res_df.loc[non_zero_mask, feature_columns])

    return res_df


def scale_chem_features(chem_descriptors_df):
    '''
    method which takes pd.DataFrame() object that represents the SMILES chemical descriptors data frame,
    and scale its non-binary features using stand scaler - scale each feature to have mean 0 and std 1.
    keep the data distribution as prior to scaling & less sensitive to outliers than scalers such as min-max

    arg:
        - chem_descriptors_df - pd.DataFrame() object with SMILES column and 194 rdkit chemical descriptors features.
    return:
        - scaled_chem_descriptors_d - pd.DataFrame() object which represent the scaled data frame
    '''
    
    # identify non-binary columns (those with more than two unique values)
    non_binary_columns = chem_descriptors_df.columns[chem_descriptors_df.nunique() > 2]
    
    # initialize the StandardScaler & copy original data frame
    scaler = StandardScaler()
    scaled_chem_descriptors_df = chem_descriptors_df.copy()
    
    # scale the non-binary columns
    scaled_chem_descriptors_df[non_binary_columns] = scaler.fit_transform(scaled_chem_descriptors_df[non_binary_columns])

    return scaled_chem_descriptors_df
#################################################################################################

def validate_smiles(smi):
    '''
    method that validates single SMILES before extracting
    rdkit chemical descriptors from it
    '''
    mol = Chem.MolFromSmiles(smi)
    return mol is not None
    


def clean(smiles_features_df):
    '''
    method that takes smiles features data frame and fills nulls values with zero

    args:
        smiles_features_df - pd.DataFrmae() object 
    return:
        smiles_features_df - pd.DataFrmae() object 
    '''
    n_nulls = smiles_features_df.isna().sum().sum()
    
    if n_nulls > 0:
        logger.info(f'There are -> {n_nulls} nulls records in smiles features data frame, will be removed')
        # replace nulls values with -> 0
        smiles_features_df = smiles_features_df.fillna(0)
        logger.info(f'after filling nulls with zeros, there are -> {smiles_features_df.isna().sum().sum()} nulls')
   
    return smiles_features_df
  #################################################################################################  