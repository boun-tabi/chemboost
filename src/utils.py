import argparse
import json
import itertools

from src.representation_models import SWX, X8mer, Random8mer, SWRandom
from src.representation_models import SW8mer, ProtVec8mer, ProtVecBPE
from src.representation_models import All8mer, SB8mer, SBBPE
from src.representation_models import SB8merDB, SWSB8mer, SWSB8merDB


def parse_terminal_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Choose kiba or bdb', choices=['bdb', 'kiba'])

    model_names = ['sw_x', 'x_8mer', 'sw_random', 'random_8mer',
                   'sw_8mer', 'pv_8mer', 'pv_bpe',
                   'all_8mer', 'sb_8mer', 'sb_bpe',
                   'sb_8mer_db', 'sw_sb_8mer', 'sw_sb_8mer_db']
    parser.add_argument('--model', help='Choose the model to train', choices=model_names)
    parser.add_argument('--savefile', help='Set name for saving')
    args = parser.parse_args()
    return args.dataset, args.model, args.savefile


def get_repr_model(dataset, model, configs):
    with open(configs[f'{dataset}_ligands']) as f:
        ligands = json.load(f)

    with open(configs[f'{dataset}_proteins']) as f:
        proteins = json.load(f)

    sb_threshold = configs[f'{dataset}_sb_threshold']

    if model == 'sw_x':
        representation_model = SWX(configs[f'{dataset}_prot_sim'])
    elif model == 'x_8mer':
        representation_model = X8mer(ligands, configs['lingo2vec'])
    elif model == 'sw_random':
        representation_model = SWRandom(configs[f'{dataset}_prot_sim'], ligands)
    elif model == 'random_8mer':
        representation_model = Random8mer(proteins, ligands, configs['lingo2vec'])
    elif model == 'sw_8mer':
        representation_model = SW8mer(configs[f'{dataset}_prot_sim'], ligands, configs['lingo2vec'])
    elif model == 'pv_8mer':
        representation_model = ProtVec8mer(proteins, ligands, configs['prot2vec'], configs['lingo2vec'])
    elif model == 'pv_bpe':
        representation_model = ProtVecBPE(proteins, ligands, configs['prot2vec'], configs['bpe2vec'], configs['smiles_bpe'])
    elif model == 'all_8mer':
        representation_model = All8mer(ligands, configs['lingo2vec'])
    elif model == 'sb_8mer':
        representation_model = SB8mer(ligands, configs['lingo2vec'], sb_threshold)
    elif model == 'sb_bpe':
        representation_model = SBBPE(ligands, configs['bpe2vec'], configs['smiles_bpe'], sb_threshold)
    elif model == 'sb_8mer_db':
        representation_model = SB8merDB(ligands, configs['lingo2vec'], sb_threshold, configs['sb_bindingdb_path'])
    elif model == 'sw_sb_8mer':
        representation_model = SWSB8mer(configs[f'{dataset}_prot_sim'], ligands, configs['lingo2vec'], sb_threshold)
    elif model == 'sw_sb_8mer_db':
        representation_model = SWSB8merDB(configs[f'{dataset}_prot_sim'], ligands, configs['lingo2vec'], sb_threshold, configs['sb_bindingdb_path'])

    return representation_model


def dict_cartesian_product(dct):
    return [dict(zip(dct.keys(), items)) for items in itertools.product(*dct.values())]
