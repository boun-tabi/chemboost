import json
import numpy as np
import pandas as pd
from src.representation_models.chemboost import ChemBoost
from src.representation_models.utils import get_lingos, get_bpe_words, flatten, read_binding_db
from src.representation_models.utils import vectorize_word_list, vectorize_molecules


class LigandCentric(ChemBoost):
    def __init__(self, word2vec_path):
        ChemBoost.__init__(self, word2vec_path)
        # Will be provided later
        self.train = None
        # Will be computed based on train data
        self.prot2vec = None

    def set_train(self, train):
        self.train = train
        self.prot2vec = self._compute_prot_vectors()


class LigandCentric8mer(LigandCentric):
    def __init__(self, ligands, lingo2vec_path):
        LigandCentric.__init__(self, lingo2vec_path)
        print('Vectorizing ligands')
        self.ligand2vec = vectorize_molecules(lambda x: get_lingos(x, q=8),
                                              ligands,
                                              self.word2vec)


class All8mer(LigandCentric8mer):
    def _compute_prot_vectors(self):
        print('All8mer: Computing protein vectors')
        prot_ids = self.train['prot_id'].unique()
        prot2vec = {}
        for prot_id in prot_ids:
            known_ligands = self.train.query('prot_id == @prot_id')['smiles']
            known_lingos_by_ligand = [get_lingos(smiles, 8) for smiles in known_ligands]
            prot2vec[prot_id] = vectorize_word_list(flatten(known_lingos_by_ligand), self.word2vec)

        return prot2vec


class SB8mer(LigandCentric8mer):
    def __init__(self, ligands, lingo2vec_path, sb_threshold):
        LigandCentric8mer.__init__(self, ligands, lingo2vec_path)
        self.sb_threshold = sb_threshold

    def _compute_prot_vectors(self):
        print('SB8mer: Computing protein vectors')
        prot_ids = self.train['prot_id'].unique()
        prot2vec = {}
        for prot_id in prot_ids:
            sb_ligands = self.train.query('prot_id == @prot_id and affinity_score > @self.sb_threshold')['smiles']
            if len(sb_ligands) == 0:
                sb_ligands = self.train.query('prot_id == @prot_id')['smiles']

            sb_lingos_by_ligand = [get_lingos(smiles, 8) for smiles in sb_ligands]
            prot2vec[prot_id] = vectorize_word_list(flatten(sb_lingos_by_ligand), self.word2vec)

        print('SB8mer: Done computing protein vectors')
        return prot2vec


class SBBPE(LigandCentric):
    def __init__(self, ligands, bpe_word2vec_path, bpe_model_path, sb_threshold):
        LigandCentric.__init__(self, bpe_word2vec_path)
        print('SBBPE: Vectorizing ligands')
        self.ligand2vec = vectorize_molecules(lambda x: get_bpe_words(x, bpe_model_path),
                                              ligands,
                                              self.word2vec)
        self.sb_threshold = sb_threshold
        self.bpe_model_path = bpe_model_path

    def _compute_prot_vectors(self):
        print('SBBPE: Computing protein vectors')
        prot_ids = self.train['prot_id'].unique()
        prot2vec = {}
        for prot_id in prot_ids:
            sb_ligands = self.train.query('prot_id == @prot_id and affinity_score > @self.sb_threshold')['smiles']
            if len(sb_ligands) == 0:
                # print(f'SBBPE: No SB ligand for {prot_id}')
                sb_ligands = self.train.query('prot_id == @prot_id')['smiles']

            sb_words_by_ligand = [get_bpe_words(smiles, self.bpe_model_path) for smiles in sb_ligands]
            prot2vec[prot_id] = vectorize_word_list(flatten(sb_words_by_ligand), self.word2vec)

        print('SBBPE: Done computing protein vectors')
        return prot2vec


class SB8merDB(LigandCentric8mer):
    def __init__(self, ligands, lingo2vec_path, sb_threshold, sb_bindingdb_path):
        LigandCentric8mer.__init__(self, ligands, lingo2vec_path)
        self.sb_threshold = sb_threshold
        self.sb_bindingdb_path = sb_bindingdb_path

    def _compute_prot_vectors(self):
        print('SB8merDB: Computing protein vectors')
        sb_binding_db = read_binding_db(self.sb_bindingdb_path, str(self.train['ligand_id'].values[0]))

        prot2vec = {}
        prot_ids = self.train['prot_id'].unique()
        merged_sb_db = pd.concat([sb_binding_db, self.train.query('affinity_score > @self.sb_threshold')])
        for prot_id in prot_ids:
            sb_ligands = merged_sb_db.query('prot_id == @prot_id')['smiles']
            if len(sb_ligands) == 0:
                sb_ligands = self.train.query('prot_id == @prot_id')['smiles']

            sb_lingos_by_ligand = [get_lingos(smiles, 8) for smiles in sb_ligands]
            prot2vec[prot_id] = vectorize_word_list(flatten(sb_lingos_by_ligand), self.word2vec)

        print('SB8merDB: Done computing protein vectors')
        return prot2vec


class SWSB8mer(LigandCentric8mer):
    def __init__(self, prot_sim_vectors_path, ligands, lingo2vec_path, sb_threshold):
        LigandCentric8mer.__init__(self, ligands, lingo2vec_path)
        self.sb_threshold = sb_threshold
        with open(prot_sim_vectors_path) as f:
            self.prot2sw = json.load(f)

    def _compute_prot_vectors(self):
        print('SWSB8mer: Computing protein vectors')
        prot_ids = self.train['prot_id'].unique()
        prot2vec = {}
        for prot_id in prot_ids:
            sb_ligands = self.train.query('prot_id == @prot_id and affinity_score > @self.sb_threshold')['smiles']
            if len(sb_ligands) == 0:
                sb_ligands = self.train.query('prot_id == @prot_id')['smiles']

            sb_lingos_by_ligand = [get_lingos(smiles, 8) for smiles in sb_ligands]
            prot2vec[prot_id] = np.hstack([self.prot2sw[prot_id],
                                           vectorize_word_list(flatten(sb_lingos_by_ligand), self.word2vec)])

        print('SWSB8mer: Done computing protein vectors')
        return prot2vec


class SWSB8merDB(LigandCentric8mer):
    def __init__(self, prot_sim_vectors_path, ligands, lingo2vec_path, sb_threshold, sb_binding_db):
        LigandCentric8mer.__init__(self, ligands, lingo2vec_path)
        self.sb_threshold = sb_threshold
        self.sb_binding_db = sb_binding_db
        with open(prot_sim_vectors_path) as f:
            self.prot2sw = json.load(f)

    def _compute_prot_vectors(self):
        print('SWSB8merDB: Computing protein vectors')
        sb_binding_db = read_binding_db(self.sb_binding_db, str(self.train['ligand_id'].values[0]))
        prot2vec = {}
        prot_ids = self.train['prot_id'].unique()
        merged_sb_db = pd.concat([sb_binding_db, self.train.query('affinity_score > @self.sb_threshold')])
        for prot_id in prot_ids:
            sb_ligands = merged_sb_db.query('prot_id == @prot_id')['smiles']
            if len(sb_ligands) == 0:
                sb_ligands = self.train.query('prot_id == @prot_id')['smiles']

            sb_lingos_by_ligand = [get_lingos(smiles, 8) for smiles in sb_ligands]
            prot2vec[prot_id] = np.hstack([self.prot2sw[prot_id],
                                           vectorize_word_list(flatten(sb_lingos_by_ligand), self.word2vec)])

        print('SWSB8merDB: Done computing protein vectors')
        return prot2vec
