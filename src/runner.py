import warnings
import time
import os
import joblib
import json
import pandas as pd

from src.training import hyper_param_tuning, test_models
from src.utils import parse_terminal_arguments, get_repr_model, dict_cartesian_product
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
start = time.time()

dataset, model, save_name = parse_terminal_arguments()

model_save_path = f'./results/{dataset}/{save_name}/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

with open('configs.json') as f:
    configs = json.load(f)

with open('xgb_params.json') as f:
    xgb_params = json.load(f)
#%%
sb_threshold = configs[f'{dataset}_sb_threshold']
train_folds_path = configs[f'{dataset}_folds']
train_folds = [pd.read_csv(f'{train_folds_path}fold_{fold_idx}.csv') for fold_idx in range(5)]
for fold in train_folds:
    fold['ligand_id'] = fold['ligand_id'].astype(str)
test = pd.read_csv(configs[f'{dataset}_test'])
test['ligand_id'] = test['ligand_id'].astype(str)

print('Read the training/test data')
representation_model = get_repr_model(dataset, model, configs)

n_phases = len(xgb_params['search_params'])
fixed_params = xgb_params['fixed_params']
best_params = {}
for phase in range(n_phases):
    print(f'Fine-tuning. Phase: {phase+1}')
    param_combinations = dict_cartesian_product(xgb_params['search_params'][phase])
    fixed_params = {**fixed_params, **best_params}
    best_models, best_params, cv_scores = hyper_param_tuning(fixed_params,
                                                             param_combinations,
                                                             representation_model,
                                                             train_folds,
                                                             sb_threshold,
                                                             model_save_path)

    cv_scores.to_csv(f'{model_save_path}cv_scores_p{phase+1}.csv', index=None)

#%%
joblib.dump(best_models, model_save_path + 'models.pkl ', compress=3)

with open(model_save_path + 'best_params.json', 'w') as f:
    json.dump({**fixed_params, **best_params}, f)

print('Done tuning. Testing...')
test_models(best_models, representation_model, train_folds, test, sb_threshold, model_save_path)
print('DONE!')

elapsed_total_time = time.time() - start
total_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_total_time))
print(f'Whole program took {total_time}')
