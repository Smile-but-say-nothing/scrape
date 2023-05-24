import numpy as np
import pandas as pd
import glob

df_caption = pd.read_excel('./92例刮痧患者体质测评数据.xlsx', sheet_name='Sheet2')
path_image = glob.glob('./Sha/*/*-*')
for idx in range(len(df_caption)):
    name_idx = [i for i, path in enumerate(path_image) if df_caption.iloc[idx, 1] in path]
    caption_list = list(filter(lambda x: x != [np.nan], [[df_caption.iloc[idx, i]] for i in range(2, 6 + 2)]))
    # assert len(name_idx) == len(caption_list), f"{len(name_idx)}, {len(caption_list)}, {df_caption.iloc[idx, 1]} 出错"
    for i, path_idx in enumerate(name_idx):
        file = open(path_image[path_idx] + '\\caption2.txt', 'w')
        print(path_image[path_idx] + '\\caption2.txt' + ' 已保存！')
        content = ''.join(caption_list[i])
        
        content = content.split('\n')[2]
        content = content.replace('痧象：', '')
        
        file.write(content)
        file.close()
