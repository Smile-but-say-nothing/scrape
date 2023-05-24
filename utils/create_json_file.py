import json
import h5py
import glob
from PIL import Image
import os
import random
import jieba


random.seed(42)
jpg_files = glob.glob('./Sha/*/*/背部一.jpg')
print(jpg_files)
random_index = []
while len(random_index) < len(jpg_files):
    x = random.randint(0, len(jpg_files) - 1)
    if x not in random_index:
        random_index.append(x)
    else:
        continue

json_data = {"info": {"description": "Sha Dataset", "version": "1.0", "time": "2023/02/28"},
             "images": [],
             "annotations": []}
counter = {'Train': 0, 'Val': 0, 'Test': 0}
for idx, file in enumerate(jpg_files):
    image = Image.open(file)
    images = {'file_path': os.path.split(file)[0].replace('/', '\\'),
              'file_name': os.path.split(file)[1],
              'index': file.split('\\')[2].split(' ')[0],
              'is_segmented': True if '_seg' in file else False,
              'height': image.height,
              'width': image.width,
              'data_captured': file.split('\\')[2].split(' ')[1]}
    json_data['images'].append(images)

    sentences = []
    sentence = open(os.path.join(os.path.split(file)[0], 'caption.txt'), 'r', encoding='gbk').read().strip().split('\n')
    sentence = sentence[::-1]
    for s in sentence:
        tokens = jieba.lcut(s, cut_all=True)
        raw = s
        sentences.append({'tokens': tokens, 'raw': raw})
    
    labels = []
    label = open(os.path.join(os.path.split(file)[0], 'label.txt'), 'r', encoding='gbk').read().strip().split('\n')
    for l in label:
        labels.append({'constitution': l.split(' ')[0], 'score': l.split(' ')[1]})
    
    annotations = {'file_path': os.path.split(file)[0].replace('/', '\\'),
                   'file_name': os.path.split(file)[1],
                   'index': file.split('\\')[2].split(' ')[0],
                   'split': 'Train' if random_index[idx] < len(jpg_files) * 0.7 else 'Test' if random_index[idx] > len(jpg_files) * 0.9 else 'Val',
                   'sentences': sentences,
                   'labels': labels
                   }
    json_data['annotations'].append(annotations)
    counter[annotations['split']] += 1
    
    print(f"{idx}, {file} processed!")

with open('dataset_sha.json', 'w', encoding='gbk') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)
    
# {'Train': 353, 'Val': 100, 'Test': 50}
print(counter)

# dictionary = {"title":"JSON教程",
#               "author":"C语言中文网",
#               "url":"http://c.biancheng.net/",
#               "catalogue":["JSON是什么？","JSONP是什么？","JSON语法规则"],
#               3:{"title":"JSON教程","author":"C语言中文网","url":"http://c.biancheng.net/",1:["JSON是什么？","JSONP是什么？","JSON语法规则"]}}
# jsonString = json.dumps(dictionary, indent=4, ensure_ascii=False)
# 
# with open('temp.json', 'w', encoding='utf8') as f:
#     json.dump(dictionary, f, indent=2, ensure_ascii=False)
# print(jsonString)
