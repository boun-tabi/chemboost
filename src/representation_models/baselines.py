import json
from collections import defaultdict
import numpy as np

from src.representation_models import RepresentationModel
from src.representation_models.utils import vectorize_molecules, load_word2vec_embedding, get_lingos


class Baseline(RepresentationModel):
    def set_train(self, train):
        pass


class SWX(Baseline):
    def __init__(self, prot_sim_vectors_path):
        with open(prot_sim_vectors_path) as f:
            self.prot2vec = json.load(f)
        self.ligand2vec = defaultdict(list)


class X8mer(Baseline):
    def __init__(self, ligands, lingo2vec_path):
        self.prot2vec = defaultdict(list)
        print('8merX: Computing ligand vectors')
        lingo2vec = load_word2vec_embedding(lingo2vec_path)
        self.ligand2vec = vectorize_molecules(lambda x: get_lingos(x, q=8),
                                              ligands,
                                              lingo2vec)


class SWRandom(Baseline):
    def __init__(self, prot_sim_vectors_path, ligands):
        with open(prot_sim_vectors_path) as f:
            self.prot2vec = json.load(f)

        np.random.seed(1)
        self.ligand2vec = {ligand_id: np.random.rand(100).tolist() for ligand_id, smiles in ligands.items()}
        print('SWRandom: Created ligand vectors...')


class Random8mer(Baseline):
    def __init__(self, proteins, ligands, lingo2vec_path):
        sw_shape = len(proteins)

        np.random.rand(1)
        self.prot2vec = {prot_id: np.random.rand(sw_shape).tolist() for prot_id in proteins.keys()}
        print('Random8mer: Loading 8mer embeddings...')
        lingo2vec = load_word2vec_embedding(lingo2vec_path)
        self.ligand2vec = vectorize_molecules(lambda x: get_lingos(x, q=8),
                                              ligands,
                                              lingo2vec)
