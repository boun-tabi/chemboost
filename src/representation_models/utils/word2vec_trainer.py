import json
import time
import sentencepiece as spm
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.model_start = time.time()

    def on_epoch_begin(self, model):
        self.epoch_start = time.time()
        print(f'Epoch {self.epoch} start')

    def on_epoch_end(self, model):
        print(f'Epoch {self.epoch} end at {time.time() - self.epoch_start}')
        print(f'Total time from start {time.time() - self.model_start}')
        self.epoch += 1


class Word2VecTrainer:
    def __init__(self, tokenized_corpus, save_path, embedding_dim=100, window_size=20, n_cpu=2):
        self.corpus = tokenized_corpus
        self.save_path = save_path
        self.dim = embedding_dim
        self.ws = window_size
        self.n_cpu = n_cpu

    def train(self):
        model = Word2Vec(self.corpus, size=self.dim, window=self.ws, workers=self.n_cpu,
                         min_count=1, sample=1e-4, negative=5, iter=20, sg=1, hs=0,
                         callbacks=[EpochLogger()])
        model.save(f'{self.save_path}.model')
        model.wv.save_word2vec_format(f'{self.save_path}.kv')


def train_bpe_smilesvec(smiles_path, bpe_model_path, embeddings_path):
    with open(smiles_path) as f:
        smiles = [s.strip() for s in f.readlines()]

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path + '.model')

    print('Tokenizing smiles')
    tokenized_smiles = [sp.encode_as_pieces(s) for s in smiles]

    print('Training word2vec with bpe')
    w2v_trainer = Word2VecTrainer(tokenized_smiles, embeddings_path,
                                  embedding_dim=100, window_size=20, n_cpu=2)
    w2v_trainer.train()


with open('configs.json') as f:
    configs = json.load(f)

# train_bpe_smilesvec(configs['chembl'], configs['smiles_bpe'], configs['bpe2vec'])
