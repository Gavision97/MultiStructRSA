
# standard library
import os
import logging
import warnings
import pickle
import pathlib
import importlib.util

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from mordred import Calculator, descriptors

import torch
import torch.nn as nn

# transformers (ChemBERTa)
from transformers import RobertaTokenizer, RobertaModel

# project utils
from utils.tools import clean, scale_nonzero

# SimSon config (without sys.path)
SIMSON_PT_WEIGHTS = "/home/gavrilev/Thesis/src/saved_objects/simson_pretrained_best_model.pth"
SIMSON_TOKENIZER_PATH = "/home/gavrilev/Thesis/src/saved_objects/pubchem_part_tokenizer.json"
_SIMSON_DIR = pathlib.Path("/home/gavrilev/Thesis/Transfer Learning: PT Models/SimSon")

from tokenizers import Tokenizer # SimSon tokenizer

# ---- SimSon config.py ----
_config_spec = importlib.util.spec_from_file_location("simson_config", _SIMSON_DIR / "config.py")
_simson_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_simson_config)
get_Config = _simson_config.get_Config

# ---- SimSon model.py ----
_model_spec = importlib.util.spec_from_file_location("simson_model", _SIMSON_DIR / "model.py")
_simson_model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_simson_model)
Xtransformer_Encoder = _simson_model.Xtransformer_Encoder

import logging
logger = logging.getLogger(__name__)

SMILES_EMB_PATH = os.path.join("embeddings", "smiles")

def save_dict_pkl(obj, save_name):
    smi_emb_save_path = os.path.join(SMILES_EMB_PATH, f'{save_name}.pkl')
    with open(smi_emb_save_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f'Successfully saved SMILES embeddings to -> {smi_emb_save_path}')


def generate_smiles_features(mol_set, save_name="smiles_embeddings", features=['simson', 'chemberta', 'morgan_fp', 'desc'], save=False, device='cuda'):
    '''
    Generates molecule feature dictionary which maps feature name (key) -> feature pd.DataFrame object (value)
    Featrues - morgan fingerprints, chemical secriptors (from rdkit), chembera & chemprop features

    '''
    
    valid_smiles=mol_set
    smiles_features_dict = {}

    ### SMILES - SimSon features ###
    if 'simson' in features:
        logger.info(f'==========================================================================')
        logger.info('generating SimSon feature dataset ...')
        simson_smiles_df = generate_simson_embeddings_df(mol_set=mol_set)
        simson_smiles_df = clean(simson_smiles_df)
        logger.info(f'finished & saved - simson features dataframe shape -> {simson_smiles_df.shape}')
        smiles_features_dict['simson'] = simson_smiles_df
    ### SMILES - chemBERTa features ###
    if 'chemberta' in features:
        logger.info(f'==========================================================================')
        logger.info('generating chemBERTa feature dataset ...')
        chemberta_model = ChemBERTaPT(smiles=valid_smiles, batch_size = 32, device=device) # initialize pre-trained chemBERTa model (with custom changes)
        chemberta_smiles_df = chemberta_model.batch_encode()
        chemberta_smiles_df = clean(chemberta_smiles_df) # check for nulls & fix (fill with 0)
        #chemberta_smiles_df.to_csv(os.path.join(dir_path, f'{name}_chemberta_smiles_df.csv'), index=False)
        logger.info(f'finished & saved - chemberta features dataframe shape -> {chemberta_smiles_df.shape}')
        smiles_features_dict['chemberta'] = chemberta_smiles_df
    
    ### SMILES - morgan fingerprints - binary 0/1 1024 features ###
    if 'morgan_fp' in features:
        logger.info(f'==========================================================================')
        logger.info('generating morgan fingerprints feature dataset ...')
        morgan_fingerprints_smiles_df = generate_morgan_fingerprints(valid_smiles)
        morgan_fingerprints_smiles_df = clean(morgan_fingerprints_smiles_df) # check for nulls & fix (fill with 0)
        #morgan_fingerprints_smiles_df.to_csv(os.path.join(dir_path, f'{name}_morgan_fingerprints_smiles_df.csv'), index=False)
        logger.info(f'finished & saved - morgan fingerprints features dataframe shape -> {morgan_fingerprints_smiles_df.shape}')
        smiles_features_dict['morgan fingerprints'] = morgan_fingerprints_smiles_df
    
    ### SMILES - Mordred chemical descriptors  ###
    if 'desc' in features:
        logger.info(f'==========================================================================')
        logger.info('generating Mordred chemical descriptors feature dataset ...')
        mordred_descriptors_smiles = calc_mordred_desc(valid_smiles, ignore_3D=True)
        mordred_descriptors_smiles_df = clean(mordred_descriptors_smiles[0])
        smiles_features_dict['mordred descriptors'] = mordred_descriptors_smiles_df
        logger.info(f'finished & saved - Mordred descriptors features dataframe shape -> {mordred_descriptors_smiles_df.shape}')

    logger.info(f'finished generating all {len(list(smiles_features_dict.keys()))} smiles features dataset successfully!')

    # save feature dictionary as pickle file
    if save:
        save_dict_pkl(smiles_features_dict, save_name)

    return smiles_features_dict


###################################################################################################

## ChemBERTa Embeddings ##
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

## Mordred Chemical Descriptors Embeddigns ##

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

## Morgan Fingerprints Embeddings ##

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

## SimSon Embeddings ##

def load_simson_encoder(
    model_path=SIMSON_PT_WEIGHTS,
    tokenizer_path=SIMSON_TOKENIZER_PATH,
    device="cpu"
):
    device = device if torch.cuda.is_available() else "cpu"

    # same as SimSom default arguments in their parser object
    args = get_Config()
    args.dic_size = 300
    args.max_len = 512
    args.d_model = 768
    args.nlayers = 2
    args.nhead = 8

    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = Xtransformer_Encoder(args).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, args, device


def smiles_to_tensor(smiles_list, tokenizer, device):
    token_ids = []

    for smi in smiles_list:
        ids = tokenizer.encode(smi).ids
        token_ids.append(ids)

    x = torch.tensor(token_ids, dtype=torch.long, device=device)
    return x

def generate_simson_embeddings_df(mol_set):
    """
    Generate SimSon embeddings dataframe from a list of SMILES strings.

    Parameters
    ----------
    mol_set : list of str
        List of SMILES strings.

    Returns
    -------
    pd.DataFrame
        DataFrame where the first column is 'SMILES' and the remaining
        columns are SimSon embedding features.
    """
    emb = extract_simson_embeddings(mol_set)   # shape: (N, D)
    emb_np = emb.numpy()

    feature_cols = [f"simson_{i}" for i in range(emb_np.shape[1])]
    simson_df = pd.DataFrame(emb_np, columns=feature_cols)
    simson_df.insert(0, "SMILES", mol_set)

    return simson_df

def extract_simson_embeddings(smiles_list, batch_size=32, device="cpu"):
    model, tokenizer, args, device = load_simson_encoder(device=device)

    all_embs = []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[start:start + batch_size]

            x = smiles_to_tensor(
                smiles_list=batch_smiles,
                tokenizer=tokenizer,
                device=device
            )

            emb = model(x)
            all_embs.append(emb.cpu())

            del x, emb
            if device == "cuda": # free chace mem
                torch.cuda.empty_cache()

    return torch.cat(all_embs, dim=0)

def extract_simson_embeddings_(smiles_list):
    model, tokenizer, args, device = load_simson_encoder()

    x = smiles_to_tensor(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        device=device
    )
    
    with torch.no_grad():
        emb = model(x)
        
    return emb.cpu()