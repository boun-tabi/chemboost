import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_mss_of_interactions(configs, dataset):
    def mss(row, train, prot_sims):
        lig_id, prot_id = row['ligand_id'], row['prot_id']
        train_prots = train.query('ligand_id == @lig_id')['prot_id']
        if len(train_prots) == 0:
            return 0  # no known interaction of this ligand in the training
        return prot_sims.loc[prot_id, train_prots].max()

    train_folds_path = configs[f'{dataset}_folds']
    train_folds = [pd.read_csv(f'{train_folds_path}fold_{fold_idx}.csv') for fold_idx in range(5)]
    k = len(train_folds)
    test = pd.read_csv(configs[f'{dataset}_test'])
    prot_sims = pd.read_csv(configs[f'{dataset}_prot_sim_matrix'], index_col=0)
    mss_scores = {}
    print('Computing mss of interactions')
    for val_index in tqdm(range(k)):
        train = pd.concat(train_folds[:val_index] + train_folds[val_index + 1:])
        mss_scores[f'mss{val_index}'] = test.apply(mss, axis=1, train=train, prot_sims=prot_sims)

    df_mss = pd.DataFrame(mss_scores)
    df_mss.to_csv(f'./data/{dataset}/mss_scores.csv', index=None)


def compute_test_mse(configs, dataset):
    models = ['sw_x', 'x_8mer', 'sw_random', 'random_8mer',
              'sw_8mer', 'pv_8mer', 'pv_bpe',
              'all_8mer', 'sb_8mer', 'sb_bpe',
              'sb_8mer_db', 'sw_sb_8mer', 'sw_sb_8mer_db',
              'deepdta']

    train_folds_path = configs[f'{dataset}_folds']
    train_folds = [pd.read_csv(f'{train_folds_path}fold_{fold_idx}.csv') for fold_idx in range(5)]
    test = pd.read_csv(configs[f'{dataset}_test'])
    df_mss = pd.read_csv(f'./data/{dataset}/mss_scores.csv')
    print('Computing test set MSE of models')
    for model in tqdm(models):
        test_preds = pd.read_csv(f'./results/{dataset}/{model}/test_predictions.csv',
                                 index_col=0, header=0,
                                 names=['pred0', 'pred1', 'pred2', 'pred3', 'pred4'])
        errors = pd.concat([test, test_preds, df_mss], axis=1)

        for i in range(len(train_folds)):
            errors[f'se{i}'] = (errors[f'pred{i}'] - errors['affinity_score']) ** 2

        errors.to_csv(f'./results/{dataset}/{model}/test_error_analysis.csv', index=None)


def analyze_mss_vs_mse(dataset, models, window_limits):
    mss_cols = [f'mss{ix}' for ix in range(5)]
    mse_cols = [f'se{ix}' for ix in range(5)]

    df_mss_vs_mse_mean_std, df_mse_by_mss_interval = {}, {}
    for model in models.keys():
        test_errors = pd.read_csv(f'./results/{dataset}/{model}/test_error_analysis.csv')
        mss_vs_mse_mean_std = []
        for ix, (mss_col, mse_col) in enumerate(zip(mss_cols, mse_cols)):
            df_windowed = test_errors.copy()
            mss_values = df_windowed[mss_col]
            df_windowed['window'] = pd.cut(mss_values, bins=window_limits, include_lowest=True)

            mse_by_mss = df_windowed.groupby('window')[mse_col].agg(['mean'])
            mss_vs_mse_mean_std.append(mse_by_mss['mean'].values)

        mss_vs_mse_mean_std = np.vstack(mss_vs_mse_mean_std)
        df_mse_by_mss_interval[model] = pd.DataFrame(mss_vs_mse_mean_std,
                                                     index=[f'fold{ix}' for ix in range(5)],
                                                     columns=[f'int{ix}' for ix in range(len(window_limits) - 1)])
        mean, std = mss_vs_mse_mean_std.mean(axis=0), mss_vs_mse_mean_std.std(axis=0)
        df_mss_vs_mse_mean_std[f'{model}_mean'] = mean
        df_mss_vs_mse_mean_std[f'{model}_std'] = std

    return pd.DataFrame(df_mss_vs_mse_mean_std), df_mse_by_mss_interval


def plot_mss_vs_mse(models, window_limits, df_mss_vs_mse):
    plt.rcParams.update({'font.size': 15})
    bar_width = 0.2
    labels = [f'({(window_limits[i] * 100):.1f}-{(window_limits[i + 1] * 100):.1f}]' for i in range(len(window_limits) - 1)]
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for ix, (model, pretty) in enumerate(models.items()):
        ax.bar(x + (ix - len(models) / 2) * bar_width, df_mss_vs_mse[f'{model}_mean'],
               bar_width, label=pretty, yerr=df_mss_vs_mse[f'{model}_std'])

    ax.set_ylabel('MSE')
    ax.set_xlabel('MSS Interval (x$10^{-2}$)')
    ax.set_title(f'{dataset.upper()}')
    ax.set_ylim(0, df_mss_vs_mse.max().max() + 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(framealpha=0.25)

    fig.tight_layout()
    fig.savefig(f'./scripts/files/{dataset}_mss', dpi=400)
    plt.show()


def mss_interval_ttest(dataset, model1, model2, df_mse_by_mss_interval, interval_ix):
    def get_scores(model):
        return df_mse_by_mss_interval[dataset][model].iloc[:, interval_ix]

    scores1, scores2 = get_scores(model1), get_scores(model2)
    t_stat, p_value = ttest_rel(scores1, scores2)

    print(f't-stat: {t_stat}\np_value: {p_value}\nconfidence: {1-p_value}')
#%%


with open('configs.json') as f:
    configs = json.load(f)

models = {'sw_8mer': 'Model (1): SW - 8mer',
          'sb_8mer_db': 'Model (7): BindingDB SB - 8mer',
          'sw_sb_8mer_db': 'Model (9): SW & BindingDB SB - 8mer',
          'deepdta': 'DeepDTA'
          }

window_limits = [0.00, 0.25, 0.4, 0.60, 1.00]
df_mse_by_mss_interval_by_dataset = {}
for dataset in ['bdb', 'kiba']:
    # compute_mss_of_interactions(configs, dataset)  # saves mss scores to file
    # compute_test_mse(configs, dataset)  # saves test mse to file
    df_mss_vs_mse_mean_std, df_mse_by_mss_interval = analyze_mss_vs_mse(dataset, models, window_limits)
    df_mse_by_mss_interval_by_dataset[dataset] = df_mse_by_mss_interval
    # plot_mss_vs_mse(models, window_limits, df_mss_vs_mse_mean_std)
#%%
dataset = 'kiba'
model1, model2 = 'sw_sb_8mer_db', 'sw_8mer'
interval_ix = 3

mss_interval_ttest(dataset, model1, model2, df_mse_by_mss_interval_by_dataset, interval_ix)
