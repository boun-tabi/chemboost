import pandas as pd
import json

models = ['sw_x', 'x_8mer', 'sw_random', 'random_8mer',
          'sw_8mer', 'pv_8mer', 'pv_bpe',
          'all_8mer', 'sb_8mer', 'sb_bpe',
          'sb_8mer_db', 'sw_sb_8mer', 'sw_sb_8mer_db']

writer = pd.ExcelWriter('./files/test_results.xlsx', engine='xlsxwriter')
for dataset in ['bdb', 'kiba']:
    results = {}
    for model in models:
        with open(f'../results/{dataset}/{model}/test_results.json') as f:
            results[model] = json.load(f)

    df_res = pd.DataFrame(results).T
    df_res.index.name = 'Model'

    workbook = writer.book
    sheetname = dataset.upper()
    worksheet = workbook.add_worksheet(sheetname)
    writer.sheets[sheetname] = worksheet
    df_res.to_excel(writer, sheet_name=sheetname)

writer.save()
