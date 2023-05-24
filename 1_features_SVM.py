import os.path
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import json
import pandas as pd
import math
from collections import Counter

# Get npy files
df = pd.read_excel('./特征计算结果/纹理计算结果/纹理计算结果-全背-平均.xlsx', sheet_name='Sheet1')
df2 = pd.read_excel('./特征计算结果/颜色计算结果/颜色计算结果-全背-平均.xlsx', sheet_name='Sheet1')
json_root = './dataset_sha.json'
with open(json_root, 'r', encoding='gbk') as f:
    data = json.load(f)

itol = {0: '平和质', 1: '气虚质', 2: '阳虚质', 3: '阴虚质', 4: '痰湿质', 5: '湿热质', 6: '血瘀质', 7: '气郁质', 8: '特禀质'}
ltoi = {'平和质': 0, '气虚质': 1, '阳虚质': 2, '阴虚质': 3, '痰湿质': 4, '湿热质': 5, '血瘀质': 6, '气郁质': 7, '特禀质': 8}
X_train, X_test, y_train, y_test = [], [], [], []
for e_a in data['annotations']:
    id = os.path.split(e_a['file_path'])[1].split(' ')[0]
    name = e_a['file_path'].split('\\')[2]
    if int(id) in df['采集编号'].values.tolist():
        features_texture = df.loc[df['采集编号'] == int(id)].values.tolist()[0][7:]
        features_color = df2.loc[df2['采集编号'] == int(id)].values.tolist()[0][7:]
    else:
        name_id = list(zip(*[d[2:4] for d in df.loc[df['姓名'] == name].values.tolist()]))
        for offset in range(1, 4):
            if int(id) + offset in name_id[1]:
                features_texture = df.loc[df['采集编号'] == int(id) + offset].values.tolist()[0][7:]
                features_color = df2.loc[df2['采集编号'] == int(id) + offset].values.tolist()[0][7:]
                break
        else:
            raise Exception
    features = features_texture + features_color
    assert len(features) == 147, 'missing features!'
    # Get constitution label
    e_list = []
    max_idx, max_value = 0, 0.0
    for e_l in e_a['labels']:
        if e_l['constitution'] == '平和质':
            continue
        if float(e_l['score']) > max_value:
            max_value = float(e_l['score'])
            max_idx = ltoi[e_l['constitution']]
    label = max_idx
    if e_a['split'] in ['Train', 'Val']:
        X_train.append(features)
        y_train.append(label)
    if e_a['split'] in ['Test']:
        X_test.append(features)
        y_test.append(label)

X_train, X_test, y_train, y_test = np.array(X_train, dtype=float), np.array(X_test, dtype=float), np.array(y_train, dtype=int), np.array(y_test, dtype=int)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# StandardScaler
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# PCA
X = np.vstack([X_train, X_test])
PCA = PCA(n_components=88)
X = PCA.fit_transform(X)
# Reindex
X_train, X_test = X[:X_train.shape[0], :], X[X_train.shape[0]:, :]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
np.save('./save/data/X_train.npy', X_train)
np.save('./save/data/X_test.npy', X_test)
np.save('./save/data/y_train.npy', y_train)
np.save('./save/data/y_test.npy', y_test)


X_train = np.load('./save/data/X_train.npy')
X_test = np.load('./save/data/X_test.npy')
y_train = np.load('./save/data/y_train.npy')
y_test = np.load('./save/data/y_test.npy')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
counter = Counter(y_test)
print(counter)

predictor = svm.SVC(C=1.0, decision_function_shape='ovr', kernel='poly', verbose=True, degree=2)
predictor.fit(X_train, y_train)
result = predictor.predict(X_test)
accuracy, precision, recall, F1 = accuracy_score(result, y_test), precision_score(result, y_test, average='weighted'), recall_score(result, y_test, average='weighted', zero_division=1), f1_score(result, y_test, average='weighted')
print(f'accuracy {accuracy:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1 {F1:.4f}')
# accuracy 0.5000, precision 0.5822, recall 0.5000, F1 0.5174
