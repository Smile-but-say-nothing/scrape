import os.path
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import numpy as np
import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from radiomics.glcm import RadiomicsGLCM
import SimpleITK as sitk
from radiomics import featureextractor
import radiomics.glszm


def cv_show(img, name='img'):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 300, 300)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def normalize(image, oldmin=0, oldmax=255, newmin=0, newmax=7):
#     normalized_image = np.round((image - oldmin) * ((newmax - newmin) / (oldmax - oldmin)) - newmin)
#     return normalized_image.astype(np.uint8)
# 
# # degree = 0
# def getGLRLM(image, level=8):
#     run_length = max(image.shape)
#     degree0Matrix = np.zeros([level, run_length])
#     counter = 0
#     for y in range(image.shape[0]):
#         for x in range(image.shape[1]):
#             nowVal = image[y][x]
#             if x + 1 >= image.shape[1]:
#                 nextVal = None
#             else:
#                 nextVal = image[y][x + 1]
#             
#             if nextVal != nowVal and counter == 0:
#                 degree0Matrix[int(nowVal)][counter] += 1
#             elif nextVal == nowVal:
#                 counter += 1
#             elif nextVal != nowVal and counter != 0:
#                 degree0Matrix[int(nowVal)][counter] += 1
#                 counter = 0
# 
#     glrlm = np.array(degree0Matrix, dtype=np.float64)
#     Ng, Nr = glrlm.shape
#     j2 = np.arange(1, Nr + 1) ** 2
#     # compute SRE
#     SRE = np.sum(glrlm / j2) / np.sum(glrlm)
#     # compute LRE
#     LRE = np.sum(glrlm * j2) / np.sum(glrlm)
#     # compute LGLRE
#     i2 = np.expand_dims(np.arange(1, Ng + 1) ** 2, axis=1)
#     LGLRE = np.sum(glrlm / i2) / np.sum(glrlm)
#     # compute GLN
#     GLN = np.sum(np.sum(glrlm, axis=1) ** 2) / np.sum(glrlm)
#     return [SRE, LRE, LGLRE, GLN]
# 


# compute frequency
# folder_paths = glob.glob('./Sha/*/*-*')
folder_paths = glob.glob('./Dataset - ALL/*/*-*')
fq = {}
for folder_path in folder_paths:
    name = folder_path.split('\\')[1]
    if name not in fq.keys():
        fq[name] = 1
    else:
        fq[name] += 1
print(fq)
print(len(fq.keys()))

type_dict = {'entire': '全背', 'top': '上', 'median': '中', 'bottom': '下',
             'left-top': '左上', 'median-top': '中上', 'right-top': '右上',
             'left-median': '左中', 'median-median': '中中', 'right-median': '右中',
             'left-bottom': '左下', 'median-bottom': '中下', 'right-bottom': '右下'}


def get_st_ed(type, shape):
    # shape: (H, W)
    h_st, h_ed, w_st, w_ed = 0, 0, 0, 0
    if type == 'entire':
        h_st, h_ed = 0, shape[0]
        w_st, w_ed = 0, shape[1]
    elif type == 'top':
        h_st, h_ed = 0, shape[0] // 3
        w_st, w_ed = 0, shape[1]
    elif type == 'median':
        h_st, h_ed = shape[0] // 3, shape[0] - shape[0] // 3
        w_st, w_ed = 0, shape[1]
    elif type == 'bottom':
        h_st, h_ed = shape[0] - shape[0] // 3, shape[0]
        w_st, w_ed = 0, shape[1]
    
    if 'left-' in type:
        w_st, w_ed = 0, shape[1] // 3
    if 'median-' in type:
        w_st, w_ed = shape[1] // 3, shape[1] - shape[1] // 3
    if 'right-' in type:
        w_st, w_ed = shape[1] - shape[1] // 3, shape[1]
    if '-top' in type:
        h_st, h_ed = 0, shape[0] // 3
    if '-median' in type:
        h_st, h_ed = shape[0] // 3, shape[0] - shape[0] // 3
    if '-bottom' in type:
        h_st, h_ed = shape[0] - shape[0] // 3, shape[0]
    return h_st, h_ed, w_st, w_ed


def compute_first_order_texture_features(paths, type='entire'):
    columns = []
    features = {}
    # df = pd.read_excel('./分析.xlsx', sheet_name='Sheet1')
    df, df_blend = None, None
    for idx, path in enumerate(paths):
        print(f"type: {type_dict[type]}, idx: {idx} / {len(paths) - 1}, path: {path}")
        file_path, image_name = os.path.split(path)
        columns.extend(['路径', '姓名', '采集编号', '图片名', '类型', '刮痧次数'])
        features.update({'路径': file_path, '姓名': path.split('\\')[1], '采集编号': int(path.split('\\')[2].split(' ')[0]), '图片名': image_name, '类型': type_dict[type], '刮痧次数': fq[path.split('\\')[1]]})
        # imread
        image_gray = cv2.imdecode(np.fromfile(path, dtype=np.int8), cv2.IMREAD_GRAYSCALE)
        h_st, h_ed, w_st, w_ed = get_st_ed(type, image_gray.shape)
        image_gray = image_gray[h_st:h_ed, w_st:w_ed]
        # cv_show(image_gray)
        # return 0
        # image_gray = cv2.resize(image_gray, (256, 256))[..., np.newaxis]
        print(f"image_gray shape: {image_gray.shape}")
        mask = cv2.imdecode(np.fromfile(path.replace('seg.jpg', 'mask.jpg'), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) // 255
        mask = mask[h_st:h_ed, w_st:w_ed]
        print(f"mask shape: {mask.shape}")
        # the mask is 0 or 1 all i.e. nothing is segmented
        mask[0, 0] = 0
        # convert to sitk object
        image_sitk = sitk.GetImageFromArray(image_gray)
        mask_sitk = sitk.GetImageFromArray(mask)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        # disable all features
        extractor.disableAllFeatures()
        
        # enable features by name
        # enabledFeatures = {'firstorder': ['Energy', 'Entropy', 'Minimum', '10Percentile', '90Percentile',
        #                                   'Maximum', 'Mean', 'Median', 'InterquartileRange', 'RootMeanSquared',
        #                                   'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'],
        #                    'glcm': ['Autocorrelation', 'Contrast', 'Correlation', 'JointEntropy', 'Idm'],
        #                    'glszm': ['SmallAreaEmphasis', 'LargeAreaEmphasis', 'GrayLevelNonUniformity', 'ZonePercentage', 'SmallAreaHighGrayLevelEmphasis'],
        #                    'glrlm': ['ShortRunEmphasis', 'LongRunEmphasis', 'GrayLevelNonUniformity', 'LowGrayLevelRunEmphasis'],
        #                    'ngtdm': ['Coarseness', 'Contrast'],
        #                    'gldm': ['SmallDependenceEmphasis', 'LargeDependenceEmphasis', 'GrayLevelNonUniformity']
        #                    }
        # extractor.enableFeaturesByName(**enabledFeatures)
        
        # 93 first order + texture features
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('ngtdm')
        extractor.enableFeatureClassByName('gldm')
        result = extractor.execute(image_sitk, mask_sitk)
        columns.extend(list(result.keys())[22:])
        if df is None:
            # create df
            df = pd.DataFrame(columns=columns, index=[x for x in range(len(paths))])
            df_blend = pd.DataFrame(columns=columns)
        # record features to df
        for key, value in result.items():
            features[str(key)] = value
        for col in df.columns.values:
            df.loc[idx, col] = float(features[col]) if not isinstance(features[col], str) else features[col]
        # blend
        if '背部二_seg.jpg' == features['图片名']:
            res_blend = np.zeros([len(df.columns.values) - 6], dtype=np.float64)
            for i in range(3):
                res_blend += np.array(df.loc[idx - i].values[6:].tolist(), dtype=np.float64)
            res_blend /= 3
            res_blend = res_blend.tolist()
            for e in [fq[path.split('\\')[1]], type_dict[type], '平均_seg', int(path.split('\\')[2].split(' ')[0]), path.split('\\')[1], file_path]:
                res_blend.insert(0, e)
            df_blend.loc[len(df_blend)] = res_blend
    
    df.to_excel('./特征计算结果/纹理计算结果/纹理计算结果-' + type_dict[type] + '.xlsx', sheet_name='Sheet1')
    df_blend.to_excel('./特征计算结果/纹理计算结果/纹理计算结果-' + type_dict[type] + '-平均.xlsx', sheet_name='Sheet1')
    print(f"dataframe of type: {type_dict[type]} saved! \n")


def basicCompute(slice, mask):
    slice = slice[mask == 1]
    # OpenCV: (H, W)
    # compute mean
    mean = np.mean(slice)
    # compute var
    var = np.var(slice)
    # get hist
    hist = cv2.calcHist([slice], [0], None, [256], [0, 256])
    hist = np.squeeze(hist)
    # compute kurt
    s = pd.Series(hist)
    hist_kurt = s.kurt()
    # compute var
    hist_var = np.var(hist)
    # compute std
    hist_std = np.std(hist)
    # compute energy
    hist_energy = np.sum(hist ** 2)
    return [mean, var, hist_kurt, hist_var, hist_std, hist_energy]


def compute_color_features(paths, type='entire'):
    columns = []
    df, df_blend = None, None
    for idx, path in enumerate(paths):
        print(f"type: {type_dict[type]}, idx: {idx} / {len(paths) - 1}, path: {path}")
        res = []
        file_path, image_name = os.path.split(path)
        res.extend([file_path, path.split('\\')[1], int(path.split('\\')[2].split(' ')[0]), image_name, type_dict[type], fq[path.split('\\')[1]]])
        columns.extend(['路径', '姓名', '采集编号', '图片名', '类型', '刮痧次数'])
        columns.extend([s + '_' + f for s in ['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B'] for f in ['mean', 'var', 'hist_kurt', 'hist_var', 'hist_std', 'hist_energy']])
        if df is None:
            # create df
            df = pd.DataFrame(columns=columns, index=[x for x in range(len(paths))])
            df_blend = pd.DataFrame(columns=columns)
        # imread
        image_rgb = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h_st, h_ed, w_st, w_ed = get_st_ed(type, image_rgb.shape)
        image_rgb = image_rgb[h_st:h_ed, w_st:w_ed, :]
        print(f"image_rgb shape: {image_rgb.shape}")
        # mask = cv2.threshold(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY)[1]
        mask = cv2.imdecode(np.fromfile(path.replace('seg.jpg', 'mask.jpg'), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) // 255
        mask = mask[h_st:h_ed, w_st:w_ed]
        print(f"mask shape: {mask.shape}")
        for i in range(3):
            # BGR
            res_slice = basicCompute(image_rgb[:, :, 2 - i], mask)
            res.extend(res_slice)
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)
        for i in range(3):
            # HSV
            res_slice = basicCompute(image_hsv[:, :, i], mask)
            res.extend(res_slice)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2LAB)
        for i in range(3):
            # LAB
            res_slice = basicCompute(image_lab[:, :, i], mask)
            res.extend(res_slice)
        df.loc[idx] = res
        # blend
        if '背部二_seg.jpg' == res[3]:
            res_blend = np.zeros([len(res) - 6], dtype=np.float64)
            for i in range(3):
                res_blend += np.array(df.loc[idx - i].values[6:].tolist(), dtype=np.float64)
            res_blend /= 3
            res_blend = res_blend.tolist()
            for e in [fq[path.split('\\')[1]], type_dict[type], '平均_seg', int(path.split('\\')[2].split(' ')[0]), path.split('\\')[1], file_path]:
                res_blend.insert(0, e)
            df_blend.loc[len(df_blend)] = res_blend

    df.to_excel('./特征计算结果/颜色计算结果/颜色计算结果-' + type_dict[type] + '.xlsx', sheet_name='Sheet1')
    df_blend.to_excel('./特征计算结果/颜色计算结果/颜色计算结果-' + type_dict[type] + '-平均.xlsx', sheet_name='Sheet1')
    print(f"dataframe of type: {type_dict[type]} saved! \n")


if __name__ == '__main__':
    # paths = glob.glob('./Sha/*/*/*_seg.jpg')
    paths = glob.glob('./Dataset - ALL/*/*/*_seg.jpg')
    print(f"len: {len(paths)}")
    for type in list(type_dict.keys())[4:]:
        compute_first_order_texture_features(paths, type=type)
        # compute_color_features(paths, type=type)

# def get_hsv(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h = [e for e in hsv[..., 0].reshape(-1) if e != 0]
#     s = [e for e in hsv[..., 1].reshape(-1) if e != 0]
#     v = [e for e in hsv[..., 2].reshape(-1) if e != 0]
# 
#     h_max, h_min, h_mean = np.max(h), np.min(h), np.mean(h)
#     s_max, s_min, s_mean = np.max(s), np.min(s), np.mean(s)
#     v_max, v_min, v_mean = np.max(v), np.min(v), np.mean(v)
#     
#     return [h_max, h_min, h_mean, s_max, s_min, s_mean, v_max, v_min, v_mean]
# 
# path_list = glob.glob('./*/*.jpg')
# last_name = path_list[0].split('\\')[1]
# hsv_list = []
# res = []
# name = None
# for path in path_list:
#     name = path.split('\\')[1]
#     print(name)
#     if name == last_name:
#         image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#         hsv_list += get_hsv(image)
#         print(hsv_list)
#     else:
#         hsv_list.insert(0, last_name)
#         res.append(hsv_list)
#         hsv_list = []
#         image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
#         hsv_list += get_hsv(image)
#         last_name = name
# hsv_list.insert(0, name)
# res.append(hsv_list)
# func = lambda x: list(map(lambda e: float(e) if not isinstance(e, str) else e, x))
# res = list(map(func, res))
# df = pd.DataFrame(res)
# df.to_excel('temp.xlsx')

# df = pd.read_excel('./temp.xlsx', sheet_name='S').fillna(0)
# print(df)
# # df.shape[0]
# for row in range(df.shape[0]):
#     x, y = [], []
#     for col in range(df.shape[1]):
#         if df.iat[row, col] != 0:
#             x.append(df.columns.values[col])
#             y.append(df.iat[row, col])
#     plt.plot(x, y)
# plt.xlabel('times')
# plt.ylabel('S_mean')
# plt.grid()
# 
# # 
# plt.savefig('S_mean_all.png')
# plt.show()
