import os
from mmcif.io.IoAdapterCore import IoAdapterCore
from Bio.PDB import MMCIFParser, PDBIO 



def cif_to_pdb(af3_dir_path):
    '''
    Convert AlphaFold3 '#.cif' prediction files to '#.pdb'
    
    :param af3_dir_path: Path to AlphaFold3 prediction files. Directory structure assumed:
    af3_dir_path/
        rna_seq_1/
            fold_rna_seq_1_model_1.cif
            fold_rna_seq_1_model_2.cif
            ...
        rna_seq_2/
            ...
    '''
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    # iterate over all RNA AF3 structure prediction files #
    for rna_pred_dir in os.listdir(af3_dir_path):
        af3_structure_predictions = [f'fold_{rna_pred_dir}_model_{i}' for i in range(0,5)]
        curr_path = os.path.join(af3_dir_path, rna_pred_dir)
        for model in af3_structure_predictions:
            curr_model_cif_path = os.path.join(curr_path, f'{model}.cif')
            curr_model_pdb_path = os.path.join(curr_path, f'{model}.pdb')

            if not os.path.exists(curr_model_cif_path):
                continue

            try:
                # convert AF3 rna sturcture prediction '.cif' file into '.pdb' file #
                structure = parser.get_structure(rna_pred_dir, curr_model_cif_path)
                io.set_structure(structure)
                io.save(curr_model_pdb_path)
                print(f"Converted: {rna_pred_dir} -> {os.path.basename(curr_model_pdb_path)}")

            except Exception as e:
                print(f"Failed converting {rna_pred_dir}: {e}")

def get_structure_coordinates(path_to_structure):
    """
    Print: (atom name, nucleotide, X, Y, Z) from a RNA structure (in .cif format)
    """
    try:
        io = IoAdapterCore() # mmcif io object
        data_container = io.readFile(path_to_structure)[0]

        atom_site = data_container.getObj('atom_site')
        labels = atom_site.getAttributeList()

        # indices of required fields (atom index, nucleotide index, x, y, z)
        atom_idx = labels.index('label_atom_id')
        nuc_idx = labels.index('label_comp_id')
        x_idx = labels.index('Cartn_x')
        y_idx = labels.index('Cartn_y')
        z_idx = labels.index('Cartn_z')

        print("ATOM\tNUC\tX\tY\tZ")

        for row in atom_site.getRowList():
            atom = row[atom_idx]
            nuc = row[nuc_idx]
            x = float(row[x_idx])
            y = float(row[y_idx])
            z = float(row[z_idx])

            print(f"{atom}\t{nuc}\t{x:.3f}\t{y:.3f}\t{z:.3f}")

    except Exception as e:
        print(f"Failed investigating structure file {path_to_structure}: {e}")


def main():
    '''
    af3_dir = os.path.join('datasets', 'AlphaFold3 RNA Predicted Structures')
    cif_to_pdb(af3_dir_path=af3_dir)ring for main
    '''
    rna_seq_1_model_1_cif_path = os.path.join('datasets', 'AlphaFold3 RNA Predicted Structures','rna_seq_1', 'fold_rna_seq_1_model_0.cif')
    get_structure_coordinates(rna_seq_1_model_1_cif_path)
if __name__ == '__main__':
    main()