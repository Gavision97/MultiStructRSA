import os
import logging
import warnings
warnings.filterwarnings('ignore')

import pickle
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from sklearn.feature_selection import VarianceThreshold
import numpy as np


from mordred import Calculator, descriptors


import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaModel
from utils.tools import clean, scale_nonzero

import logging
logger = logging.getLogger(__name__)

SMILES_EMB_PATH = os.path.join("embeddings", "smiles", "smiles_embeddings.pkl")

def save_dict_pkl(obj, path=SMILES_EMB_PATH):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'Successfully saved SMILES embeddings to -> {SMILES_EMB_PATH}')


def generate_smiles_features(mol_set, save=False, device='cuda'):
    '''
    Generates molecule feature dictionary which maps feature name (key) -> feature pd.DataFrame object (value)
    Featrues - morgan fingerprints, chemical secriptors (from rdkit), chembera & chemprop features

    '''
    
    valid_smiles=mol_set
    smiles_features_dict = {}

    ### SMILES - chemBERTa features ###
    logger.info(f'==========================================================================')
    logger.info('generating chemBERTa feature dataset ...')
    chemberta_model = ChemBERTaPT(smiles=valid_smiles, batch_size = 32, device=device) # initialize pre-trained chemBERTa model (with custom changes)
    chemberta_smiles_df = chemberta_model.batch_encode()
    chemberta_smiles_df = clean(chemberta_smiles_df) # check for nulls & fix (fill with 0)
    #chemberta_smiles_df.to_csv(os.path.join(dir_path, f'{name}_chemberta_smiles_df.csv'), index=False)
    logger.info(f'finished & saved - chemberta features dataframe shape -> {chemberta_smiles_df.shape}')
    smiles_features_dict['chemberta'] = chemberta_smiles_df
    
    ### SMILES - rdkit expert features - chemical descriptors (194 features in total) ###
    logger.info(f'==========================================================================')
    logger.info('generating rdkit chemical descriptors feature dataset ...')
    chem_descriptors_ls_path = os.path.join('saved_objects', 'chemical_descriptors_list.pkl') # load list of first 194 chemical descriptors feature names
    with open(chem_descriptors_ls_path, 'rb') as f:
        chem_descriptors_ls = pickle.load(f)

    chemical_descriptors_smiles_df = generate_descriptors(valid_smiles, chem_descriptors_ls)
    chemical_descriptors_smiles_df = clean(chemical_descriptors_smiles_df) # check for nulls & fix (fill with 0)
    #chemical_descriptors_smiles_df.to_csv(os.path.join(dir_path, f'{name}_chemical_descriptors_smiles_df.csv'), index=False)
    logger.info(f'finished & saved - rdkit chemical descriptiors features dataframe shape -> {chemical_descriptors_smiles_df.shape}')   
    smiles_features_dict['chemical descriptors'] = chemical_descriptors_smiles_df
    
    ### SMILES - morgan fingerprints - binary 0/1 1024 features ###
    logger.info(f'==========================================================================')
    logger.info('generating morgan fingerprints feature dataset ...')
    morgan_fingerprints_smiles_df = generate_morgan_fingerprints(valid_smiles)
    morgan_fingerprints_smiles_df = clean(morgan_fingerprints_smiles_df) # check for nulls & fix (fill with 0)
    #morgan_fingerprints_smiles_df.to_csv(os.path.join(dir_path, f'{name}_morgan_fingerprints_smiles_df.csv'), index=False)
    logger.info(f'finished & saved - morgan fingerprints features dataframe shape -> {morgan_fingerprints_smiles_df.shape}')
    smiles_features_dict['morgan fingerprints'] = morgan_fingerprints_smiles_df
    
    ### SMILES - Mordred chemical descriptors  ###
    #logger.info(f'==========================================================================')
    #logger.info('generating Mordred chemical descriptors feature dataset ...')
    #mordred_descriptors_smiles = calc_mordred_desc(valid_smiles, ignore_3D=True)
    #mordred_descriptors_smiles_df = clean(mordred_descriptors_smiles[0])
    #smiles_features_dict['mordred descriptors'] = mordred_descriptors_smiles_df
    #logger.info(f'finished & saved - Mordred descriptors features dataframe shape -> {mordred_descriptors_smiles_df.shape}')

    logger.info('finished generating all 4 smiles features dataset successfully!')

    # save feature dictionary as pickle file
    if save:
        save_dict_pkl(smiles_features_dict)

    return smiles_features_dict


###################################################################################################


class ChemBERTaPT(nn.Module):
    def __init__(self, smiles, batch_size, device="cuda"):
        super(ChemBERTaPT, self).__init__()
        self.model_name = "DeepChem/ChemBERTa-77M-MTR"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.chemberta = RobertaModel.from_pretrained(self.model_name, add_pooling_layer=False).to(device)
        self.smiles_list = smiles
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.embedding_dim = 384

    def forward(self, input_ids, attention_mask):
        bert_output = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_embedding = bert_output.last_hidden_state[:, 0, :]
        return cls_token_embedding

    def batch_encode(self):
        self.eval()
        smiles_list = self.smiles_list
        embeddings = []

        valid_indices = []
        invalid_indices = []
        valid_smiles = []

        for i, smi in enumerate(smiles_list):
            if Chem.MolFromSmiles(smi) is None:
                invalid_indices.append(i)
            else:
                valid_indices.append(i)
                valid_smiles.append(smi)

        if len(valid_smiles) == 0:
            zero_df = pd.DataFrame(
                torch.zeros((len(smiles_list), self.embedding_dim), device=self.device).cpu().numpy(),
                columns=[f"chemberta_feature_{i}" for i in range(self.embedding_dim)]
            )
            return pd.concat([pd.DataFrame(smiles_list, columns=["SMILES"]), zero_df], axis=1)

        with torch.no_grad():
            for i in range(0, len(valid_smiles), self.batch_size):
                batch = valid_smiles[i:i + self.batch_size]
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                output_embeddings = self(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(output_embeddings)

        valid_embeddings = torch.cat(embeddings, dim=0)

        full_embeddings = torch.zeros((len(smiles_list), self.embedding_dim), device=self.device)
        valid_ptr = 0
        for i in range(len(smiles_list)):
            if i in invalid_indices:
                continue
            full_embeddings[i] = valid_embeddings[valid_ptr]
            valid_ptr += 1

        embedding_df = pd.DataFrame(full_embeddings.cpu().numpy())
        embedding_df.columns = [f"chemberta_feature_{i}" for i in range(self.embedding_dim)]
        smiles_df = pd.DataFrame(smiles_list, columns=["SMILES"])
        res_df = pd.concat([smiles_df, embedding_df], axis=1)

        #res_df = scale_nonzero_rows(res_df)
        return res_df


###################################################################################################
def preprocess_mordred_descriptors(mordred_ls, smi_ls):
    desc_df = pd.DataFrame(mordred_ls)
    desc_df.insert(0, "SMILES", list(smi_ls))

    # numeric stability: coerce non-numeric (e.g., Missing) -> NaN, then fill
    desc_df.iloc[:, 1:] = (
        desc_df.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    # drop columns with constant values (only descriptors, keep SMILES)
    desc_cols = desc_df.columns[1:]
    keep_nonconst = desc_df[desc_cols].nunique() > 1
    desc_df = pd.concat([desc_df[["SMILES"]], desc_df.loc[:, desc_cols[keep_nonconst]]], axis=1)

    # variance threshold (only descriptors)
    selector = VarianceThreshold(threshold=1e-2)
    reduced_descriptors = selector.fit_transform(desc_df.iloc[:, 1:])
    keep_cols = desc_df.iloc[:, 1:].columns[selector.get_support()]

    desc_df = pd.DataFrame(reduced_descriptors, columns=keep_cols)
    desc_df.insert(0, "SMILES", list(smi_ls))

    return desc_df

def calc_mordred_desc(smi_ls, ignore_3D=True):
    calculator = Calculator(descriptors, ignore_3D=ignore_3D)
    print(len(calculator.descriptors))

    mols = [Chem.MolFromSmiles(smi) for smi in smi_ls]
    descriptors_ls = []

    desc_names = [str(d) for d in calculator.descriptors]

    for mol in mols:
        if mol is not None:
            result = calculator(mol)
            desc_dict = {str(k): v for k, v in result.asdict().items()}
        else:
            desc_dict = {name: 0.0 for name in desc_names}

        descriptors_ls.append(desc_dict)

    desc_df = preprocess_mordred_descriptors(descriptors_ls, smi_ls)

    desc_map = {
        smi: desc_df.loc[i, desc_df.columns[1:]].values.astype(np.float32)
        for i, smi in enumerate(smi_ls)
    }

    return desc_df, desc_map

##################################################################################################
def generate_descriptors(smiles, all_descriptors_list):
    # Split descriptor list
    first_ten_descriptors = all_descriptors_list[:10]
    descriptors_list = all_descriptors_list[10:]
    descriptor_values = []

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)

        if mol:
            # Calculate first 10 core descriptors
            initial_descriptors = [
                AllChem.CalcNumAtomStereoCenters(mol),
                AllChem.CalcNumAmideBonds(mol),
                Descriptors.MolWt(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.FractionCSP3(mol)
            ]

            # Compute the rest safely, fallback = 0
            extra_desc = []
            for descriptor in descriptors_list:
                try:
                    if hasattr(Descriptors, descriptor):
                        val = getattr(Descriptors, descriptor)(mol)
                        # handle None or NaN explicitly
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            val = 0
                    else:
                        val = 0
                except Exception as e:
                    print(f"[Warning] Descriptor '{descriptor}' failed for SMILES '{smile}': {e}")
                    val = 0
                extra_desc.append(val)

            all_descriptors = initial_descriptors + extra_desc

        else:
            # Invalid SMILES → vector of zeros
            all_descriptors = [0] * len(all_descriptors_list)

        descriptor_values.append(all_descriptors)

    # Create DataFrame with correct column names
    feature_df = pd.DataFrame(descriptor_values, columns=first_ten_descriptors + descriptors_list)

    '''
    # Scale if enough samples
    if feature_df.shape[0] > 2:
        feature_df = scale_chem_features(feature_df)
    '''
    
    # Return SMILES + features
    chemical_descriptors_smiles_df = pd.concat(
        [pd.Series(smiles, name='SMILES'), feature_df],
        axis=1
    )
    
    chemical_descriptors_smiles_df = scale_nonzero(chemical_descriptors_smiles_df)
    return chemical_descriptors_smiles_df



def generate_descriptors_(smiles, all_descriptors_list):

    # extract first 10 descriptors
    first_ten_descriptors = all_descriptors_list[:10]

    descriptors_list = all_descriptors_list[10:]
    descriptor_values = []

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        
        # if the molecule is valid, calculate the descriptors (already checked for validity beofre ..)
        if mol:
            initial_descriptors = [
                AllChem.CalcNumAtomStereoCenters(mol),
                AllChem.CalcNumAmideBonds(mol),
                Descriptors.MolWt(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.FractionCSP3(mol)
            ]
            all_descriptors = initial_descriptors + [getattr(Descriptors, descriptor)(mol) if hasattr(Descriptors, descriptor) else None for descriptor in descriptors_list]
            descriptor_values.append(all_descriptors)
        else:
            # in case mol is invalid, it will get zero embedding
            descriptor_values.append([0] * len(descriptors_list))
    
    feature_df = pd.DataFrame(descriptor_values, columns=first_ten_descriptors + descriptors_list)
    

    '''
        if feature_df.shape[0] > 2:
        # scale non-binary features using standart scaler (in order to scale & keep feature distribution)
        feature_df = scale_chem_features(feature_df)
    '''

    # return smiles column back to scaled data frame
    chemical_descriptors_smiles_df = pd.concat([pd.Series(smiles, name='SMILES'), feature_df], axis=1)
    
    return chemical_descriptors_smiles_df
    
###################################################################################################

def generate_morgan_fingerprints(smiles_list, radius=4, nbits=1024):
    fingerprint_data = []
    
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        
        if mol:
            # generate the Morgan fingerprint for the molecule
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            
            # convert the fingerprint to a list of 0s and 1s
            fingerprint_list = list(fingerprint)
            fingerprint_data.append(fingerprint_list)
        else:
            fingerprint_data.append([0] * nbits)
    
    morgan_df = pd.DataFrame(fingerprint_data, columns=[f'MorganFP_{i}' for i in range(nbits)])
    
    morgan_df['SMILES'] = smiles_list
    morgan_df = morgan_df[['SMILES'] + [f'MorganFP_{i}' for i in range(nbits)]] # reorder columns
    
    return morgan_df
    
###################################################################################################

