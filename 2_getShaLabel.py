import pandas as pd
import numpy as np
import glob
import os

label_txt_path = glob.glob('./Sha/*/*/label.txt')
print(label_txt_path)
for path in label_txt_path:
    os.remove(path)

df = pd.read_excel('./92例刮痧患者体质测评数据.xlsx', sheet_name='Sheet1')
constitution = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
path_image = glob.glob('./Sha/*/*-*')
for i in range(1, len(df)):
    name = df.iloc[i, 1]
    path_included = [p for p in path_image if name in p]
    for idx, j in enumerate(range(0, 60, 10)):
        if idx >= len(path_included):
            print(i, name, idx, '无图片')
            break
        score = df.iloc[i, 12 + j:21 + j].tolist()
        if np.nan in score:
            print(i, name, idx, '无Label')
            break
        try:
            # score = [0 if s < 40 else s for s in score]
            score = score
        except TypeError:
            print(i, name, idx, score)
        # score = [bool(s) for s in score]
        print(i, name, idx, score, '成功')
        # res = list(map(lambda x: x[0] * x[1], zip(score, constitution)))
        res = list(map(lambda x: [x[0], str(x[1])], zip(constitution, score)))
        # res = [e for e in res if e != '']
        res = [e for e in res if '' not in e]
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [' '.join(e) for e in res]
        res = '\n'.join(res)
        # assert idx < len(path_included), f"i: {i}, name: {name}, idx: {idx}"
        file = open(path_included[idx] + '\\label.txt', 'w')
        file.write(res)
        file.close()
        
