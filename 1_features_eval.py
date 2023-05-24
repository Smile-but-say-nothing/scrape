import pandas as pd
import pymannkendall as mk

df = pd.read_excel('./纹理计算结果-平均.xlsx', sheet_name='刮痧次数6')
df_mk = pd.DataFrame(columns=df.columns.values)
for idx, i in enumerate(range(0, len(df), 6)):
    df_mk.loc[idx, ['路径', '图片名', '刮痧次数']] = df.loc[i, ['路径', '图片名', '刮痧次数']]
    # loop of features
    for f in df.columns.values[3:]:
        data_mk = df.loc[i:i + 5, f].tolist()
        res = mk.original_test(data_mk, alpha=0.1)
        df_mk.loc[idx, f] = str(res.trend) + ' ' + str(res.p)
df_mk.to_excel('./纹理计算结果-平均-MK检验.xlsx', sheet_name='Sheet1')
exit()
