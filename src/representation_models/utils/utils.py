import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import sentencepiece as spm


def read_binding_db(path, ligand_id):
    if 'CHEMBL' in ligand_id:
        drop_col, use_col = 'cid', 'chembl_id'
    else:
        drop_col, use_col = 'chembl_id', 'cid'
        
    return pd.read_csv(path).drop(columns=[drop_col]).rename(
                        {use_col: 'ligand_id'}, axis='columns').dropna(
                        subset=['ligand_id'], axis='rows')

def flatten(word_list_by_ligand):
    return [word for ligand in word_list_by_ligand for word in ligand]

def load_word2vec_embedding(embeddings_path):
    return KeyedVectors.load_word2vec_format(embeddings_path + '.kv', binary=False)

def vectorize_word_list(tokens, token2vec):
    embedding_dim = token2vec.vector_size
    token_vecs = []
    for token in tokens:
        try:
            token_vecs.append(token2vec[token])
        except KeyError:
            token_vecs.append([0.] * embedding_dim)
    
    return np.array(token_vecs).mean(axis=0)

def vectorize_molecules(tokenizer_func, molecules, word2vec):
    mols_to_words = {mol_id: tokenizer_func(mol_str) for mol_id, mol_str in molecules.items()}
    return {mol_id: vectorize_word_list(words, word2vec) for mol_id, words in mols_to_words.items()}

def get_kmers(sequence, k=3):
    return [sequence[i: i + 3] for i in range(0, len(sequence), 3)]

def get_bpe_words(smiles, bpe_model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path + '.model')
    return sp.encode_as_pieces(smiles)

def get_lingos(smiles, q=8):
    placeholders = ['D', 'E', 'J', 'X', 'j', 't', 'z', 'x', 'd', 'R']
    two_letter_elements = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe',
            'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Tc', 'Ru', 'Rh',
            'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
            'Dy', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
            'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Bk', 'Es', 'Fm', 'Md',
            'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']

    def insert_placeholders(smiles):
        el_to_placeholder = {}
        placeholder_count = 0
        for el in two_letter_elements:
            if el in smiles:
                placeholder = placeholders[placeholder_count]
                el_to_placeholder[el] = placeholder
                placeholder_count = placeholder_count + 1
                smiles = smiles.replace(el, placeholder)
        
        return smiles, el_to_placeholder

    smiles, mappings = insert_placeholders(smiles)
    lingos = [smiles[ix: ix + q] for ix in range(len(smiles) - (q - 1))]
    def remove_placeholders(lingo, el_to_placeholder):
        for element, placeholder in el_to_placeholder.items():
            lingo = lingo.replace(placeholder, element)
        return lingo
    lingos = [remove_placeholders(lingo, mappings) for lingo in lingos]
    return lingos
