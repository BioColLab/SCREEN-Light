import numpy as np
import pandas as pd
import sys
from transformers import T5EncoderModel, T5Tokenizer
import gc
import re
import h5py

from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import pickle
import argparse
import os
import pandas as pd
from Bio import SeqIO
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
import esm
from torch_geometric.utils import degree, add_self_loops, subgraph, to_undirected, remove_self_loops, coalesce

import math


def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    # Optionally, download locally.
    #model = T5EncoderModel.from_pretrained("/home/pantong/Code/ProtT5/").to(device)
    model = model.to(dtype=torch.float16)

    model.half()
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    #Optionally, download locally.
    #tokenizer = T5Tokenizer.from_pretrained("/home/pantong/Code/ProtT5/", do_lower_case=False, force_download=True)

    return model, tokenizer


def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, max_residues=4000, max_seq_len=1000,
                   max_batch=100):
    results = {"residue_embs": dict(), "protein_embs": dict()}

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq

        if len(seq) > 1000:
            seq = seq[:500] + seq[-500:]

        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(passed_time / 60, avg_time))
    print('\n############# END #############')
    return results


def ESM_init(seq):
    result_dict = {}
    model_location = "esm2_t33_650M_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    seq_feat = seq_feature(seq)
    token_repr, contact_map_proba, logits = esm_extract(model, batch_converter, seq, layer=33, approach='last',
                                                        dim=1280)

    assert len(contact_map_proba) == len(seq)
    edge_index, edge_weight = contact_map(contact_map_proba)

    result_dict = {
        'seq': seq,
        'seq_feat': torch.from_numpy(seq_feat),
        'token_representation': token_repr.half(),
        'num_nodes': len(seq),
        'num_pos': torch.arange(len(seq)).reshape(-1, 1),
        'edge_index': edge_index,
        'edge_weight': edge_weight}
    return result_dict


def protein_init(Sequence):
    result_dict = {}
    model_location = "esm2_t33_650M_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()

    seqs = dict()
    for seq in Sequence:
        if len(seq) > 1000:
            seq_max = seq[:1000]  # + seq[-500:]
        else:
            seq_max = seq
        seqs[seq] = seq_max
    model_T5, tokenizer_T5 = get_T5_model()
    results = get_embeddings(model_T5, tokenizer_T5, seqs, "True", "True")

    for seq in tqdm(Sequence):
        if len(seq) > 1000:
            seq_max = seq[:1000] #+ seq[-500:]
        else:
            seq_max = seq

        res_feature = results["residue_embs"][seq_max][:].astype(np.float32)
        pro_feature = results["protein_embs"][seq_max][:].astype(np.float32)

        token_repr, contact_map_proba, logits = esm_extract(model, batch_converter, seq_max, layer=33, approach='last',dim=1280)
        contact_adj = (contact_map_proba >= 0.5)

        assert len(contact_map_proba) == len(seq_max)
        edge_index, edge_weight = contact_map(contact_map_proba)

        result_dict[seq] = {
            'seq': seq_max,
            'seq_feat': res_feature,
            'pro_feat': pro_feature,
            'num_nodes': len(seq_max),
            'contact_map':contact_adj,
            'num_pos': torch.arange(len(seq_max)).reshape(-1, 1),
            'edge_index': edge_index,
            'edge_weight': edge_weight}

    return result_dict


# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    if 'U' in pro_seq or 'B' in pro_seq or 'Z' in pro_seq:
        print('U or B or Z in Sequence')
    pro_seq = pro_seq.replace('U', 'X').replace('B', 'X').replace("O", "X").replace("Z", "X")
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):
        # print("111111",pro_seq[i])
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])

    res_fea = cal_atomfea(pro_seq)
    return np.concatenate((pro_hot, pro_property, res_fea), axis=1)


def contact_map(contact_map_proba, contact_threshold=0.5):
    num_residues = contact_map_proba.shape[0]
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_map_proba[row, col].float()
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues) + 1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:], (torch.arange(num_residues) - 1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues) + 2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:], (torch.arange(num_residues) - 2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)

    return edge_index, edge_weight


def esm_extract(model, batch_converter, seq, layer=36, approach='mean', dim=2560):
    pro_id = 'A'
    if len(seq) <= 700:
        data = []
        data.append((pro_id, seq))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

        logits = results["logits"][0].cpu().numpy()[1: len(seq) + 1]
        contact_prob_map = results["contacts"][0].cpu().numpy()
        token_representation = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
        assert token_representation.size(0) == layer

        if approach == 'last':
            token_representation = token_representation[-1]
        elif approach == 'sum':
            token_representation = token_representation.sum(dim=0)
        elif approach == 'mean':
            token_representation = token_representation.mean(dim=0)

        token_representation = token_representation.cpu().numpy()
        token_representation = token_representation[1: len(seq) + 1]
    else:
        contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
        token_representation = np.zeros((len(seq), dim))
        logits = np.zeros((len(seq), layer))
        interval = 350
        i = math.ceil(len(seq) / interval)
        # ======================
        # =                    =
        # =                    =
        # =          ======================
        # =          =*********=          =
        # =          =*********=          =
        # ======================          =
        #            =                    =
        #            =                    =
        #            ======================
        # where * is the overlapping area
        # subsection seq contact map prediction
        for s in range(i):
            start = s * interval  # sub seq predict start
            end = min((s + 2) * interval, len(seq))  # sub seq predict end
            sub_seq_len = end - start

            # prediction
            temp_seq = seq[start:end]
            temp_data = []
            temp_data.append((pro_id, temp_seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
            batch_tokens = batch_tokens.to(next(model.parameters()).device, non_blocking=True)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[i for i in range(1, layer + 1)], return_contacts=True)

            # insert into the global contact map
            row, col = np.where(contact_prob_map[start:end, start:end] != 0)
            row = row + start
            col = col + start
            contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                0].cpu().numpy()
            contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0

            logits[start:end] += results['logits'][0].cpu().numpy()[1: len(temp_seq) + 1]
            logits[row] = logits[row] / 2.0

            ## TOKEN
            subtoken_repr = torch.cat([results['representations'][i] for i in range(1, layer + 1)])
            assert subtoken_repr.size(0) == layer
            if approach == 'last':
                subtoken_repr = subtoken_repr[-1]
            elif approach == 'sum':
                subtoken_repr = subtoken_repr.sum(dim=0)
            elif approach == 'mean':
                subtoken_repr = subtoken_repr.mean(dim=0)

            subtoken_repr = subtoken_repr.cpu().numpy()
            subtoken_repr = subtoken_repr[1: len(temp_seq) + 1]

            trow = np.where(token_representation[start:end].sum(axis=-1) != 0)[0]
            trow = trow + start
            token_representation[start:end] = token_representation[start:end] + subtoken_repr
            token_representation[trow] = token_representation[trow] / 2.0

            if end == len(seq):
                break

    return torch.from_numpy(token_representation), torch.from_numpy(contact_prob_map), torch.from_numpy(logits)


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


from Bio.PDB import PDBParser

biopython_parser = PDBParser()

one_to_three = {"A": "ALA",
                "C": "CYS",
                "D": "ASP",
                "E": "GLU",
                "F": "PHE",
                "G": "GLY",
                "H": "HIS",
                "I": "ILE",
                "K": "LYS",
                "L": "LEU",
                "M": "MET",
                "N": "ASN",
                "P": "PRO",
                "Q": "GLN",
                "R": "ARG",
                "S": "SER",
                "T": "THR",
                "V": "VAL",
                "W": "TRP",
                "Y": "TYR",
                "B": "ASX",
                "Z": "GLX",
                "X": "UNK",
                "*": " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path):
    structure = biopython_parser.get_structure('random_id', protein_path)[0]
    seq = ''
    chain_str = ''
    for i, chain in enumerate(structure):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += 'X'
                    chain_str += str(chain.id)
                    print("encountered unknown AA: ", residue.get_resname(),
                          ' in the complex. Replacing it with a dash X.')

    return seq, chain_str


restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'}


def cal_atomfea(seqence):
    all_for_assign = np.loadtxt("all_assign.txt")
    xx = seqence
    x_p = np.zeros((len(xx), 7))
    for j in range(len(xx)):
        try:
            if restype_1to3[xx[j]] == 'ALA':
                x_p[j] = all_for_assign[0, :]
            elif restype_1to3[xx[j]] == 'CYS':
                x_p[j] = all_for_assign[1, :]
            elif restype_1to3[xx[j]] == 'ASP':
                x_p[j] = all_for_assign[2, :]
            elif restype_1to3[xx[j]] == 'GLU':
                x_p[j] = all_for_assign[3, :]
            elif restype_1to3[xx[j]] == 'PHE':
                x_p[j] = all_for_assign[4, :]
            elif restype_1to3[xx[j]] == 'GLY':
                x_p[j] = all_for_assign[5, :]
            elif restype_1to3[xx[j]] == 'HIS':
                x_p[j] = all_for_assign[6, :]
            elif restype_1to3[xx[j]] == 'ILE':
                x_p[j] = all_for_assign[7, :]
            elif restype_1to3[xx[j]] == 'LYS':
                x_p[j] = all_for_assign[8, :]
            elif restype_1to3[xx[j]] == 'LEU':
                x_p[j] = all_for_assign[9, :]
            elif restype_1to3[xx[j]] == 'MET':
                x_p[j] = all_for_assign[10, :]
            elif restype_1to3[xx[j]] == 'ASN':
                x_p[j] = all_for_assign[11, :]
            elif restype_1to3[xx[j]] == 'PRO':
                x_p[j] = all_for_assign[12, :]
            elif restype_1to3[xx[j]] == 'GLN':
                x_p[j] = all_for_assign[13, :]
            elif restype_1to3[xx[j]] == 'ARG':
                x_p[j] = all_for_assign[14, :]
            elif restype_1to3[xx[j]] == 'SER':
                x_p[j] = all_for_assign[15, :]
            elif restype_1to3[xx[j]] == 'THR':
                x_p[j] = all_for_assign[16, :]
            elif restype_1to3[xx[j]] == 'VAL':
                x_p[j] = all_for_assign[17, :]
            elif restype_1to3[xx[j]] == 'TRP':
                x_p[j] = all_for_assign[18, :]
            elif restype_1to3[xx[j]] == 'TYR':
                x_p[j] = all_for_assign[19, :]
        except:
            print("exception residue", xx[j])
    return x_p
