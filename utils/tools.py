import os
import warnings

import pandas as pd


from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.preprocessing import StandardScaler


import logging
logger = logging.getLogger(__name__)

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
    