import time
from collections import defaultdict
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.metrics import concordance_index, mse


def create_cv_results_column_names(search_params, n_cv):
    cv_stats_list = list(search_params.keys())
    for ix in range(n_cv):
        for mode in ('train', 'val'):
            for metric in ('mse',):
                cv_stats_list.append(f'fold_{ix}_{mode}_{metric}')

    for mode in ('train', 'val'):
        for metric in ('mse',):
            for stat in ('mean', 'std'):
                cv_stats_list.append(f'{mode}_{metric}_{stat}')

    return cv_stats_list


def evaluate(gold_truths, predictions, affinity_threshold, mode='train'):
    if mode == 'train' or mode == 'val':
        return {'MSE': mse(gold_truths, predictions)}

    elif mode == 'test':
        return {'CI': concordance_index(gold_truths.tolist(), predictions.tolist()),
                'MSE': mse(gold_truths, predictions)}

    raise ValueError('Invalid evaluation mode!')


def hyper_param_tuning(fixed_params, search_params, representation_model, folds, sb_threshold, save_path):
    print('Training: Started Hyper-parameter tuning')
    n_cv = len(folds)
    print('Training: Prepared cv scores table')
    best_mse, best_xgb_models, best_params = np.inf, None, None
    all_params_stats = []
    for param_ix, params in enumerate(search_params):
        param_start = time.time()
        print(f'Training: Running CV on {params}. {param_ix + 1}/{len(search_params)}')
        # Store parameter values
        param_stats = list(params.values())
        cv_scores_train, cv_scores_val = defaultdict(list), defaultdict(list)
        xgb_params = {**fixed_params, **params}
        xgb_models = []
        for val_index in range(n_cv):
            xgb = XGBRegressor(**xgb_params)
            cv_start = time.time()
            print(f'Training: Val fold: {val_index}')
            train = pd.concat(folds[:val_index] + folds[val_index + 1:])
            val = folds[val_index]

            print('Training: Setting training set of representation model')
            representation_model.set_train(train)
            train_vecs = train.apply(lambda x: representation_model.vectorize_interaction(x), axis=1, result_type='expand')
            val_vecs = val.apply(lambda x: representation_model.vectorize_interaction(x), axis=1, result_type='expand')
            eval_set = [(train_vecs, train['affinity_score']), (val_vecs, val['affinity_score'])]
            print(f'Training: Fitting XGBoost (Val fold: {val_index})')
            xgb.fit(train_vecs, train['affinity_score'],
                    eval_set=eval_set,
                    verbose=False)
            xgb_models.append(deepcopy(xgb))

            train_predictions = xgb.predict(train_vecs)
            val_predictions = xgb.predict(val_vecs)

            print(f'Training: Evaluating predictions (Val fold: {val_index})')
            fold_scores_train = evaluate(train['affinity_score'], train_predictions, sb_threshold, mode='train')
            fold_scores_val = evaluate(val['affinity_score'], val_predictions, sb_threshold, mode='val')

            print(f'Training: Storing the results (Val fold: {val_index})')
            for metric, score in fold_scores_train.items():
                cv_scores_train[metric].append(score)

            for metric, score in fold_scores_val.items():
                cv_scores_val[metric].append(score)

            param_stats.extend([fold_scores_train['MSE']])
            param_stats.extend([fold_scores_val['MSE']])

            elapsed_cv_time = time.time() - cv_start
            cv_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_cv_time))
            print(f'Training: These folds took {cv_time}')

        for mode in (cv_scores_train, cv_scores_val):
            for metric in ('MSE',):
                for stat in (np.mean, np.std):
                    param_stats.append(stat(mode[metric]))

        cv_val_mse = np.mean(cv_scores_val['MSE'])
        if cv_val_mse < best_mse:
            print(f'Training: Best XGB is changed from {best_params} to {params}')
            best_mse = cv_val_mse
            best_xgb_models = xgb_models
            best_params = params

        all_params_stats.append(param_stats)
        elapsed_param_time = time.time() - param_start
        param_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_param_time))
        print(f'Training: This param combination took {param_time}')

    cv_scores_column_names = create_cv_results_column_names(best_params, n_cv)
    cv_scores = pd.DataFrame(all_params_stats, columns=cv_scores_column_names)
    print('Training: Done CV')
    return best_xgb_models, best_params, cv_scores


def test_models(xgb_models, representation_model, folds, test, sb_threshold, save_path):
    test_scores = defaultdict(list)
    all_test_predictions = []
    for val_index in range(len(folds)):
        xgb = xgb_models[val_index]
        print(f'Test: Ignoring fold: {val_index}')
        train = pd.concat(folds[:val_index] + folds[val_index + 1:])

        print('Test: Setting training set of representation model')
        representation_model.set_train(train)
        test_vecs = test.apply(lambda x: representation_model.vectorize_interaction(x), axis=1, result_type='expand')

        test_predictions = xgb.predict(test_vecs)
        print(f'Test: Evaluating predictions (Val fold: {val_index})')
        fold_scores = evaluate(test['affinity_score'], test_predictions, sb_threshold, mode='test')

        for stat, score in fold_scores.items():
            test_scores[stat].append(score)

        all_test_predictions.append(test_predictions)

    test_results = {}
    for metric in ('CI', 'MSE'):
        for stat, stat_name in ((np.mean, 'mean'), (np.std, 'std')):
            test_results[f'{metric}_{stat_name}'] = stat(test_scores[metric])

    print(f'Test: Test Results: {test_results}')
    print('Test: Dumping the results and predictions.')
    with open(save_path + 'test_results.json', 'w') as f:
        json.dump(test_results, f)

    pd.DataFrame(all_test_predictions).T.to_csv(save_path + 'test_predictions.csv')
