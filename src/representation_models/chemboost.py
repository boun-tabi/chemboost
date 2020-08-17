from src.representation_models import RepresentationModel
from src.representation_models.utils import load_word2vec_embedding


class ChemBoost(RepresentationModel):
    def __init__(self, word2vec_path):
        self.word2vec = load_word2vec_embedding(word2vec_path)
