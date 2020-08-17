import pandas as pd
from scipy.stats import ttest_rel
from src.metrics import mse, concordance_index


def ttest_paired(dataset, model1, model2, metric_name='mse'):
    metrics = {'mse': mse, 'ci': concordance_index}
    metric = metrics[metric_name]
    y_test = pd.read_csv(f'./data/{dataset}/test.csv', usecols=['affinity_score'])

    def get_test_scores(model):
        preds = pd.read_csv(f'./results/{dataset}/{model}/test_predictions.csv', index_col=0)
        return [metric(y_test['affinity_score'], preds.iloc[:, ix]) for ix in range(5)]

    scores1, scores2 = get_test_scores(model1), get_test_scores(model2)
    t_stat, p_value = ttest_rel(scores1, scores2)

    print(f't-stat: {t_stat}\np_value: {p_value}\nconfidence: {1-p_value}')
#%%


model1, model2 = 'deepdta', 'sw_sb_8mer_db'
#%%
print(model1, model2)
print('BDB')
ttest_paired('bdb', model1, model2, 'mse')
ttest_paired('bdb', model1, model2, 'ci')

print('KIBA')
ttest_paired('kiba', model1, model2, 'mse')
ttest_paired('kiba', model1, model2, 'ci')
