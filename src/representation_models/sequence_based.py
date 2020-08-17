import json
from src.representation_models import ChemBoost
from src.representation_models.utils import load_word2vec_embedding, vectorize_molecules
from src.representation_models.utils import get_kmers, get_lingos, get_bpe_words


class SW8mer(ChemBoost):
    def __init__(self, prot_sim_vectors_path, ligands, lingo2vec_path):
        ChemBoost.__init__(self, lingo2vec_path)
        with open(prot_sim_vectors_path) as f:
            self.prot2vec = json.load(f)

        self.ligand2vec = vectorize_molecules(lambda x: get_lingos(x, q=8),
                                              ligands,
                                              self.word2vec)
        print('SW8mer: Vectorized ligands and proteins')

    def set_train(self, train):
        pass


class ProtVec(ChemBoost):
    def __init__(self, proteins, kmer2vec_path, word2vec_path):
        ChemBoost.__init__(self, word2vec_path)
        kmer2vec = load_word2vec_embedding(kmer2vec_path)
        self.prot2vec = vectorize_molecules(lambda x: get_kmers(x, k=3),
                                            proteins,
                                            kmer2vec)

    def set_train(self, train):
        pass


class ProtVec8mer(ProtVec):
    def __init__(self, proteins, ligands, kmer2vec_path, lingo2vec_path):
        print('ProtVec8mer: Reading kmer and lingo embeddings')
        ProtVec.__init__(self, proteins, kmer2vec_path, lingo2vec_path)
        print('ProtVec8mer: Vectorizing ligands')
        self.ligand2vec = vectorize_molecules(lambda x: get_lingos(x, q=8),
                                              ligands,
                                              self.word2vec)


class ProtVecBPE(ProtVec):
    def __init__(self, proteins, ligands, kmer2vec_path, bpe2vec_path, bpe_model_path):
        ProtVec.__init__(self, proteins, kmer2vec_path, bpe2vec_path)
        print('ProtVecBPE: Vectorizing ligands')
        self.ligand2vec = vectorize_molecules(lambda x: get_bpe_words(x, bpe_model_path),
                                              ligands,
                                              self.word2vec)
