import json
import time
import sentencepiece as spm


class BPETrainer:
    def __init__(self, corpus_path, save_path, vocab_size=20000):
        self.corpus_path = corpus_path
        self.save_path = save_path
        self.vocab_size = vocab_size
        print(f'BPETrainer: Created BPE trainer. Corpus: {self.corpus_path}, Save: {self.save_path}, VS:{self.vocab_size}')

    def train(self):
        print('BPETrainer: Started training')
        train_command = f'--input={self.corpus_path} --model_prefix={self.save_path} ' + \
                        f'--vocab_size={self.vocab_size} --model_type=bpe --hard_vocab_limit=False ' + \
                        '--character_coverage=0.99 --max_sentencepiece_length=100 ' + \
                        '--split_by_number=False --split_by_unicode_script=False'
        spm.SentencePieceTrainer.Train(train_command)


with open('configs.json') as f:
    configs = json.load(f)

start = time.time()
trainer = BPETrainer(configs['chembl'], configs['smiles_bpe'], 100)
trainer.train()
print(f'Training took {time.time()} - start')
