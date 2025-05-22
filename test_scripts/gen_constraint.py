import os
import logging
import json
from itertools import repeat
import gemmi
# from openbabel import pybel
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from glob import glob
import numpy as np
from scipy.spatial import cKDTree
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import Align, SeqIO
from Bio.PDB import MMCIFParser, MMCIFIO, PDBIO, PDBParser, Polypeptide
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner, substitution_matrices

SCHRODINGER = '/data_ssd/protein/zhuwenyu/schrodinger2024-1'
PROTENIX = '/data_hdd/home/casp15/miniconda3/bin/protenix'
PROTENIX_CONSTRAINT = '/data_hdd/home/casp15/miniconda3/envs/proteinx_constraint_esm/bin/protenix'
NUM_SAMPLES = 5
CHAIN_SCHEME = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def three_to_one(three_letter_code):
    index = Polypeptide.three_to_index(three_letter_code)
    one_letter_code = Polypeptide.index_to_one(index)
    return one_letter_code


def align_sequences(seq1, seq2, open_gap_score=0, extend_gap_score=0):  # ignore gaps
    # Use BLOSUM62 matrix for scoring
    matrix = substitution_matrices.load("BLOSUM62")

    # Initialize aligner
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score

    # Perform alignment
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]  # Take the best alignment

    # Calculate similarity score
    score = best_alignment.score

    # Calculate max theoretical score for normalization
    max_score = sum(
        matrix[a, a] for a in seq1  # Assuming identical alignment for max score
    )

    # Normalize score
    normalized_score = score / max_score  # Normalize between 0 and 1

    return best_alignment, normalized_score


def seq_mapping(alignment):
    # 创建索引映射表
    seq1_to_seq2 = {}  # 映射表：seq1索引 -> seq2索引
    seq2_to_seq1 = {}  # 反向映射表：seq2索引 -> seq1索引

    # 遍历对齐片段，生成映射关系
    for (start1, end1), (start2, end2) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        for i, j in zip(range(start1, end1), range(start2, end2)):
            seq1_to_seq2[i] = j
            seq2_to_seq1[j] = i
    return seq1_to_seq2, seq2_to_seq1


def struct_clean(pdb_file, keep_chain="A"):
    st = gemmi.read_structure(pdb_file)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()  # will not remove modifications
    st.remove_empty_chains()
    if len(st[0]) > 1:
        if keep_chain not in [chain.name for chain in st[0]]:
            raise ValueError(f'Chain {keep_chain} not found in the pdb file.')
        for chain in st[0]:
            if chain.name != keep_chain:
                del st[0][chain.name]
    if len(st[0]) == 0:
        raise ValueError('No chain left after cleaning. Please specify the correct chain to keep.')
    # chain must be continuous
    last_res = None
    for res in st[0][keep_chain]:
        if last_res is not None and int(str(res.seqid)) - int(str(last_res.seqid)) > 1:
            raise NotImplementedError(
                'Incontinuous chain is splitted into two chains and cannot be handled. '
                'Please retry with fasta file specified or manually format the pdb or json file.'
            )

    new_file = pdb_file.replace('.pdb', '_clean.pdb')
    st.write_pdb(new_file)
    return new_file


def get_binding_pockets(
    pdb, lig_coord, thres=6, renumber=True
):
    mdl = PDBParser().get_structure("x", pdb)[0]
    chain_resid = []
    for cid, chain in zip(CHAIN_SCHEME, mdl):
        resid = set()
        for i, res in enumerate(chain):
            # skip het
            if res.id[0] != " ":
                continue
            res_coord = np.array(
                [i.get_coord() for i in res.get_atoms() if i.element != "H"]
            )
            dist = np.linalg.norm(
                res_coord[:, None, :] - lig_coord[None, :, :], axis=-1
            ).min()
            if dist <= thres:
                if renumber:
                    resid.add(i)
                else:
                    resid.add(str(res.id[1]))
            if renumber:
                chain_resid.extend([(cid, i) for i in resid])
            else:
                chain_resid.extend([(chain.id, i) for i in resid])
    if len(chain_resid) == 0:
        return None
    return set(chain_resid)


def fix_struct(home,prot,overwrite=False):
    
    prot_fixed = os.path.join(home,os.path.splitext(prot)[0]+'_fixed.pdb')
    os.chdir(home)
    if os.path.exists(prot_fixed) and not overwrite:
        print('Using existed fixed pdb files!')
    else:
        os.system(f'{SCHRODINGER}/utilities/structconvert {prot} {os.path.splitext(prot)[0]+".maegz"}')
        os.system(f'{SCHRODINGER}/utilities/prepwizard -j prepwizard_{os.path.splitext(prot)[0]} -watdist 5 -propka_pH 7.4 -rmsd 0.30 -HOST localhost:1 -TMPLAUNCHDIR -ATTACHED -WAIT {os.path.splitext(prot)[0]+".maegz"} {os.path.splitext(prot)[0]+"_fixed.maegz"}')
        os.system(f'{SCHRODINGER}/utilities/structconvert {prot_fixed} {prot_fixed.replace(".maegz",".pdb")}')

    if not os.path.exists('ligand_fixed.sdf') and os.path.exists('ligand.sdf'):
        os.system(f'{SCHRODINGER}/utilities/structconvert ligand.sdf ligand.maegz')
        os.system(f'{SCHRODINGER}/utilities/prepwizard -j ligand_fix -watdist 5 -propka_pH 7.4 -rmsd 0.30 -HOST localhost:1 -TMPLAUNCHDIR -ATTACHED -WAIT ligand.maegz ligand_fixed.maegz')
        os.system(f'{SCHRODINGER}/utilities/structconvert ligand_fixed.maegz ligand_fixed.sdf')

    return


def generate_json(home, receptor=None, need_fix=True, fasta_file=None):
    if fasta_file is not None:
        with open(os.path.join(home,fasta_file), 'r') as f:
            seq = SeqIO.read(f, 'fasta')
            seq = str(seq.seq)
        with open(os.path.join(home, f'{receptor.split(".")[0]}-fromfa.json'), 'w') as f:
            json.dump([{'sequences': [{'proteinChain': {'sequence': seq, 'count': 1}}], 'name': os.path.basename(receptor).split('.')[0]}], f)
        return os.path.join(home, f'{receptor.split(".")[0]}-fromfa.json')
    if need_fix:
        fix_struct(home, receptor)
        fix_flag = '_fixed'
    else:
        fix_flag = ''
    os.system(f'{PROTENIX} tojson --input {os.path.join(home, receptor.replace(".maegz",fix_flag+".pdb"))} --out_dir {home}')
    return glob(os.path.join(home, f'{receptor.split(".")[0]}-*.json'))[0]


def run_with_smiles(home, smiles, json_path, seed=[101], constraint_json=None, mas=True, esm=True):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data[0]['sequences'].append({'ligand': {'ligand': smiles, 'count': 1}})
    if constraint_json is not None:
        logger.warning('Using constraint for cofolding, which is preview version and uses a different set of parameters.')
        with open(constraint_json, 'r') as f:
            constraint = json.load(f)
        num_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
        constraint['constraint']['contact'][0]['atom2'] = int(num_atoms / 2)  # Set the atom2 to the middle of the ligand
        logger.info(f'Using constraint: {constraint}, but it is not guaranteed.')
        data[0].update(constraint)
    with open(os.path.join(home, 'input.json'), 'w') as f:
        json.dump(data, f)
    if isinstance(seed, int):
        seed = [seed]
    if constraint_json is not None:
        os.system(f'{PROTENIX_CONSTRAINT} predict --input {os.path.join(home, "input.json")} --out_dir {home} --seeds {",".join(map(str, seed))} {"--use_msa_server" if mas else ""} {"--use_esm" if esm else ""}')
    else:
        os.system(f'{PROTENIX} predict --input {os.path.join(home, "input.json")} --out_dir {home} --seeds {",".join(map(str, seed))} --use_msa_server')


def find_graph_isomorphism(G1, G2, feature_key="feature"):
    """
    Checks if two graphs are isomorphic while preserving integer node features.

    Parameters:
    G1 (nx.Graph): First graph.
    G2 (nx.Graph): Second graph.
    feature_key (str): The node attribute key to compare (default: "feature").

    Returns:
    dict or None: A mapping of G1 nodes to G2 nodes if isomorphic, otherwise None.
    """
    # Define node attribute matching function
    def node_match(n1, n2):
        return n1.get(feature_key) == n2.get(feature_key)

    # Check for isomorphism
    GM = GraphMatcher(G1, G2, node_match=node_match)
    return GM.mapping if GM.is_isomorphic() else None


def mol2G(mol):
    """
    Convert RDKit molecule to NetworkX graph.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): RDKit molecule.

    Returns:
    nx.Graph: NetworkX graph representation of the molecule.
    """
    G = nx.Graph()
    G.add_nodes_from([(i, {"feature": a.GetAtomicNum()}) for i, a in enumerate(mol.GetAtoms())])
    G.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
    return G


def split_cif(cif_file, smiles=None):
    parser = MMCIFParser()
    sd_file = cif_file.replace('.cif', '.sdf')
    pdb_file = cif_file.replace('.cif', '.pdb')
    st = parser.get_structure('x', cif_file)
    io = MMCIFIO()
    for chain in st[0]:
        for res in chain:
            if res.get_resname() == 'l01':
                break
    io.set_structure(res)
    chain.detach_child(res.id)
    io.save(sd_file + '.cif')
    # mol = next(pybel.readfile('sdf', sd_file + '.cif'))
    # mol.write('sdf', sd_file)
    os.system(f'{SCHRODINGER}/utilities/structconvert {sd_file + ".cif"} {sd_file}')
    mol = Chem.SDMolSupplier(sd_file)[0]
    mol2 = Chem.MolFromSmiles(smiles)
    G1 = mol2G(mol)
    G2 = mol2G(mol2)
    mapping = find_graph_isomorphism(G1, G2)
    if mapping is None:
        raise ValueError('No isomorphism found between the reference ligand and the predicted ligand.')
    AllChem.EmbedMolecule(mol2)
    conf1 = mol.GetConformer()
    conf2 = mol2.GetConformer()
    for i, j in mapping.items():
        conf2.SetAtomPosition(j, conf1.GetAtomPosition(i))
    with open(sd_file, 'w') as f:
        f.write(Chem.MolToMolBlock(mol2))
    io = PDBIO()
    io.set_structure(st)
    io.save(pdb_file)
    return sd_file, pdb_file


def generate_constraint_json(receptor, ref_lig, distance=10, seq_map=None):
    ref_mol = Chem.SDMolSupplier(ref_lig)[0]
    ref_coords = ref_mol.GetConformer().GetPositions()
    st = PDBParser().get_structure('x', receptor)
    dist_l = []
    if seq_map is None:
        seq_map = repeat(None, len(st[0]))
    for i, (chain, mapping) in enumerate(zip(st[0], seq_map), 1):
        for j, res in enumerate(chain):
            res_coords = np.array([a.get_coord() for a in res])
            dist = np.linalg.norm(ref_coords.reshape(-1, 1, 3) - res_coords.reshape(1, -1, 3), axis=-1).sum()
            if mapping and j not in mapping:
                continue  # 跳过未比对的残基
            
            dist_l.append((i, (mapping[j] if mapping else j) + 1, dist))  # resi mapping starts from 0, but constraint starts from 1
    dist_l = sorted(dist_l, key=lambda x: x[2])
    # with open(os.path.join(home, 'constraint.json'), 'w') as f:
    #     json.dump({'constraint': {"contact": [{'entity1': dist_l[0][0], 'copy1': 1, 'position1': dist_l[0][1], 'entity2': len(st[0])+1, 'copy2': 1, 'position2': 1, 'atom2': 0, 'max_distance': distance}]}}, f)
    
    constraint ={"contact": [{'entity1': dist_l[0][0], 'copy1': 1, 'position1': dist_l[0][1], 'entity2': len(st[0])+1, 'copy2': 1, 'position2': 1, 'atom2': 0, 'max_distance': distance}]}
    return constraint


def chamfer_distance(pc1, pc2):
    """
    Compute Chamfer Distance between two point clouds.
    
    Parameters:
    pc1: numpy array of shape (N, D) - First point cloud
    pc2: numpy array of shape (M, D) - Second point cloud

    Returns:
    float - Chamfer Distance
    """
    # Build KD-Trees for fast nearest neighbor search
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    # Compute nearest neighbor distances
    dist1, _ = tree1.query(pc2)  # Distance from each point in pc2 to nearest in pc1
    dist2, _ = tree2.query(pc1)  # Distance from each point in pc1 to nearest in pc2

    return np.mean(dist1) + np.mean(dist2)  # Symmetric Chamfer Distance


def extract_pdb_sequence(pdb_file):
    seqs = {}
    st = PDBParser().get_structure('x', pdb_file)
    model = st[0]
    for chain in model:
        residues = []
        for res in chain:
            if res.id[0] != " ":  # Skip heteroatoms
                continue
            if res.resname not in ["A", "T", "C", "G", "U", "DA", "DT", "DC", "DG", "DU"]:
                residues.append(res)
        seq = []
        for residue in residues:
            try:
                seq.append(three_to_one(residue.resname))
            except KeyError:
                logger.warning(
                    f"Unknown 3-letter amino acid code: {residue.resname} from {pdb_file}"
                )
                seq.append("X")  # Replace unknown residues with X
        if seq:
            seqs[chain.id] = "".join(seq)
    return seqs


def extract_json_sequence(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)[0]['sequences'][0]["proteinChain"]['sequence']


def cofolding(
        home,
        receptor,
        ref_lig,
        smiles,
        round_idx,
        para_idx,
        seed=101,
        sample=0,
        need_fix=False,
        constraint=True,
        constraint_distance=10,
        msa=True,  # could be False only when using esm
        esm=True,
        iou_threshold=0.3,
        chain_id=None,
        fasta_file=None,
):
    receptor_name = os.path.basename(receptor).split(".")[0]
    docking_path = os.path.join(home, f'round_{para_idx}_{round_idx}_cofolding')
    json_path = glob(os.path.join(home, f'{receptor_name}-*.json'))
    if len(json_path) == 0:
        json_path = generate_json(home, receptor, need_fix=need_fix, fasta_file=fasta_file)
    else:
        json_path = json_path[0]
    
    pdb_seq = extract_pdb_sequence(os.path.join(home, receptor))
    if len(pdb_seq) > 1:
        if chain_id is None:
            raise NotImplementedError(f'Multiple chains found in the receptor {receptor}, please specify the chain id.')
        pdb_seq = pdb_seq[chain_id]
    else:
        pdb_seq = list(pdb_seq.values())[0]
    json_seq = extract_json_sequence(json_path)
    alignment, score = align_sequences(pdb_seq, json_seq)
    Align.write(alignment, os.path.join(docking_path, 'alignment.txt'), "clustal")
    seq_map, _ = seq_mapping(alignment)
    if score < 0.9:
        logger.warning(f'Alignment score between pdb and cofolding seq is less than 0.9, {score}.')
    if constraint:
        if not os.path.exists(os.path.join(home, 'constraint.json')):
            constraint_json = generate_constraint_json(home, receptor, ref_lig, distance=constraint_distance, seq_map=[seq_map])
        else:
            constraint_json = os.path.join(home, 'constraint.json')
    else:
        constraint_json = None
    os.makedirs(docking_path, exist_ok=True)
    run_with_smiles(docking_path, smiles, json_path, constraint_json=constraint_json, seed=seed, mas=msa, esm=esm)
    if sample not in range(NUM_SAMPLES):
        ref_lig = Chem.SDMolSupplier(ref_lig)[0]
        ref_coords = ref_lig.GetConformer().GetPositions()
        ref_pocket = get_binding_pockets(os.path.join(home, receptor), ref_coords)
        ref_pocket_mapped = {(i[0], seq_map.get(i[1])) for i in ref_pocket}
        file_list = []
        pocket_list = []
        for i in range(NUM_SAMPLES):
            sd_file, pdb_file = split_cif(os.path.join(docking_path, receptor_name, f'seed_{seed}', 'predictions', f'{receptor_name}_seed_{seed}_sample_{i}.cif'), smiles=smiles)
            file_list.append((sd_file, pdb_file))
            mol = Chem.SDMolSupplier(sd_file)[0]
            coords = mol.GetConformer().GetPositions()
            pocket_list.append(get_binding_pockets(pdb_file, coords))
        print(ref_pocket, pocket_list)
        iou_list = [len(ref_pocket_mapped & pocket) / len(ref_pocket_mapped | pocket) if pocket is not None else 0 for pocket in pocket_list]
        sample = np.argmax(iou_list)
        sd_file, pdb_file = file_list[sample]
        if iou_list[sample] < iou_threshold:
            raise ValueError(f'Non of 5 cofolding samples has IoU larger than {iou_threshold} with the reference ligand. Aborted.')
        logger.info(f'No sample id is provided, using the closest sample {sample} to the reference ligand, with IoU {iou_list[sample]}.')
        # not working because protein structure is not aligned!!!
        # ref_lig = Chem.SDMolSupplier(ref_lig)[0]
        # ref_coords = ref_lig.GetConformer().GetPositions()
        # file_list = []
        # dist_list = []
        # for i in range(NUM_SAMPLES):
        #     sd_file, pdb_file = split_cif(os.path.join(docking_path, receptor_name, f'seed_{seed}', 'predictions', f'{receptor_name}_seed_{seed}_sample_{i}.cif'), smiles=smiles)
        #     file_list.append((sd_file, pdb_file))
        #     mol = Chem.SDMolSupplier(sd_file)[0]
        #     coords = mol.GetConformer().GetPositions()
        #     dist_list.append(chamfer_distance(ref_coords, coords))
        # sample = np.argmin(dist_list)
        # sd_file, pdb_file = file_list[sample]
        # logger.info(f'No sample id is provided, using the closest sample {sample} to the reference ligand, with chamfer distance {dist_list[sample]}.')
    else:
        sd_file, pdb_file = split_cif(os.path.join(docking_path, receptor_name, f'seed_{seed}', 'predictions', f'{receptor_name}_seed_{seed}_sample_{sample}.cif'), smiles=smiles)
    if not os.path.exists(sd_file):
        raise ValueError(f'Cofolding failed at round {round_idx}!')
    os.system(f'cp {pdb_file} {os.path.dirname(pdb_file)}/{receptor_name}.pdb')
    os.system(f'cp {sd_file} {os.path.dirname(sd_file)}/ligand.sdf')
    pdb_file = os.path.join(f'round_{para_idx}_{round_idx}_cofolding',receptor_name, f'seed_{seed}', 'predictions', f'{receptor_name}.pdb')
    sd_file = os.path.join(f'round_{para_idx}_{round_idx}_cofolding',receptor_name, f'seed_{seed}', 'predictions', 'ligand.sdf')
    return sd_file, pdb_file


def inplace_score(sd_file, pdb_file, method="mininplace",pose_name='cofolding'):  # method could be "inplace" or "mininplace" if do not docking
    if not os.path.exists(f'{os.path.dirname(sd_file)}/inplace_score.csv'):
        os.system(f'SCHRODINGER={SCHRODINGER} bash MedChem_R1/inplace_score.sh {sd_file} {pdb_file} {method} {pose_name}')
    with open(f'{os.path.dirname(sd_file)}/inplace_score.csv') as f:
        docking_score = float(f.read().strip())
    return docking_score



def process_constraint(json_seq, receptor_path,ref_lig):
    
    
    
    pdb_seq = extract_pdb_sequence(receptor_path)
    
    pdb_seq = list(pdb_seq.values())[0]
    #json_seq = extract_json_sequence(json_path)
    alignment, score = align_sequences(pdb_seq, json_seq)
    seq_map, _ = seq_mapping(alignment)
    #print(score)
    if score < 0.9:
        logger.warning(f'Alignment score between pdb and cofolding seq is less than 0.9, {score}.')
    #print(seq_map)
    constraint_json = generate_constraint_json(receptor_path, ref_lig, distance=10, seq_map=[seq_map])
    #print(constraint_json)
    return constraint_json
    



if __name__ == '__main__':
    home ="/home/gaobowen/ProtenixAffinity/tmp"

    receptor_path = "/home/gaobowen/ProtenixAffinity/Uni-FEP-Benchmarks-main/uni_fep_benchmarks/Merck8|SYK/processed_protein.pdb"
    ref_lig = "/home/gaobowen/ProtenixAffinity/Uni-FEP-Benchmarks-main/uni_fep_benchmarks/Merck8|SYK/ref_ligand.sdf"
    json_seq = "MASSGMADSANHLPFFFGNITREEAEDYLVQGGMSDGLYLLRQSRNYLGGFALSVAHGRKAHHYTIERELNGTYAIAGGRTHASPADLCHYHSQESDGLVCLLKKPFNRPQGVQPKTGPFEDLKENLIREYVKQTWNLQGQALEQAIISQKPQLEKLIATTAHEKMPWFHGKISREESEQIVLIGSKTNGKFLIRARDNNGSYALCLLHEGKVLHYRIDKDKTGKLSIPEGKKFDTLWQLVEHYSYKADGLLRVLTVPCQKIGTQGNVNFGGRPQLPGSHPATWSAGGIISRIKSYSFPKPGHRKSSPAQGNRQESTVSFNPYEPELAPWAADKGPQREALPMDTEVYESPYADPEEIRPKEVYLDRKLLTLEDKELGSGNFGTVKKGYYQMKKVVKTVAVKILKNEANDPALKDELLAEANVMQQLDNPYIVRMIGICEAESWMLVMEMAELGPLNKYLQQNRHVKDKNIIELVHQVSMGMKYLEESNFVHRDLAARNVLLVTQHYAKISDFGLSKALRADENYYKAQTHGKWPVKWYAPECINYYKFSSKSDVWSFGVLMWEAFSYGQKPYRGMKGSEVTAMLEKGERMGCPAGCPREMYDLMNLCWTYDVENRPGFAAVELRLRNYYYDVVN"


    constraint_dic = process_constraint(json_seq,receptor_path, ref_lig)
    