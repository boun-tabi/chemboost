import numpy as np


class RepresentationModel:
    def set_train(self, train):
        raise NotImplementedError('A representation model must have a set train function')

    def vectorize_interaction(self, interaction):
        return np.hstack([self.ligand2vec[interaction['ligand_id']],
                          self.prot2vec[interaction['prot_id']]])
