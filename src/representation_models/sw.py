import json
import numpy as np
from src.representation_models import RepresentationModel
from src.representation_models.utils import load_word2vec_embedding, word_list2vec, get_lingos


class SW(RepresentationModel):
    def __init__(self, prot_sim_vectors_path):
        with open(prot_sim_vectors_path) as f:
            self.prot2vec = json.load(f)

    def set_train(self, train):
        pass


class SWPubChem(SW):
    def __init__(self, prot_sim_vectors_path, ligand_sim_vectors_path):
        SW.__init__(self, prot_sim_vectors_path)
        with open(ligand_sim_vectors_path) as f:
            ligand_vectors = json.load(f)

        if 'CHEMBL' not in list(ligand_vectors)[0]:
            ligand_vectors = {int(lig_id): vec for lig_id, vec in ligand_vectors.items()}

        self.ligand2vec = {lig_id: vector for lig_id, vector in ligand_vectors.items()}

    def __str__(self):
        return 'sw_pubchem'


class SW8mer(SW):
    def __init__(self, prot_sim_vectors_path, ligands, lingo2vec_path):
        SW.__init__(self, prot_sim_vectors_path)
        print('SW8mer: Loading 8mer embeddings...')
        self.word2vec = load_word2vec_embedding(lingo2vec_path)
        self.ligand2vec = {ligand_id: word_list2vec(get_lingos(smiles, 8), self.word2vec) for ligand_id, smiles in ligands.items()}
        print('SW8mer: Computed ligand vectors...')

    def __str__(self):
        return 'sw_8mer'


class SWRandom(SW):
    def __init__(self, prot_sim_vectors_path, ligands):
        SW.__init__(self, prot_sim_vectors_path)
        self.ligand2vec = {ligand_id: np.random.rand(100).tolist() for ligand_id, smiles in ligands.items()}
        print('SWRandom: Created ligand vectors...')

    def __str__(self):
        return 'sw_random'


class SWOnly(SW):
    def __init__(self, prot_sim_vectors_path, ligands):
        SW.__init__(self, prot_sim_vectors_path)
        self.ligand2vec = {ligand_id: [] for ligand_id, smiles in ligands.items()}
        print('SWOnly: Created ligand vectors...')

    def __str__(self):
        return 'sw_only'


class Random8mer(SW):
    def __init__(self, prot_sim_vectors_path, ligands, lingo2vec_path):
        SW.__init__(self, prot_sim_vectors_path)
        sw_shape = len(self.prot2vec)
        prot_ids = self.prot2vec.keys()
        self.prot2vec = {prot_id: np.random.rand(sw_shape).tolist() for prot_id in prot_ids}

        print('Random8mer: Loading 8mer embeddings...')
        self.word2vec = load_word2vec_embedding(lingo2vec_path)
        self.ligand2vec = {ligand_id: word_list2vec(get_lingos(smiles, 8), self.word2vec) for ligand_id, smiles in ligands.items()}
        print('Random8mer: Computed ligand vectors...')

    def __str__(self):
        return 'random_8mer'


class Only8mer(SW):
    def __init__(self, prot_sim_vectors_path, ligands, lingo2vec_path):
        SW.__init__(self, prot_sim_vectors_path)
        prot_ids = self.prot2vec.keys()
        self.prot2vec = {prot_id: [] for prot_id in prot_ids}

        print('8mer_only: Loading 8mer embeddings...')
        self.word2vec = load_word2vec_embedding(lingo2vec_path)
        self.ligand2vec = {ligand_id: word_list2vec(get_lingos(smiles, 8), self.word2vec) for ligand_id, smiles in ligands.items()}
        print('8mer_only: Computed ligand vectors...')

    def __str__(self):
        return '8mer_only'
