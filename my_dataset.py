import PIL.Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import json
from PIL import Image, ImageOps
from lxml import etree
import pandas as pd
import spacy
import glob
import jieba
import re
from utils.clean_captions import clean_caption
from utils.build_base import Vocabulary


class SegDataset(Dataset):
    def __init__(self, root, transform_image=None, transform_mask=None):
        super(SegDataset, self).__init__()
        self.image_path = os.path.join(root, 'JPEGImages')
        self.mask_path = os.path.join(root, 'JPEGMasks')
        # self.image_list = [os.path.join(self.image_path, img) for img in os.listdir(self.image_path)]
        self.image_list = glob.glob(os.path.join(self.image_path, '*痧象*.jpg'))
        # self.mask_list = [os.path.join(self.mask_path, img) for img in os.listdir(self.image_path)]
        self.mask_list = [os.path.join(self.mask_path, os.path.split(img)[1]) for img in self.image_list]
        # self.mask_list = [os.path.join(self.mask_path, img.replace('_training.tif', '_manual1.gif')) for img in os.listdir(self.image_path)]
        # check masks
        for img in self.mask_list:
            assert os.path.exists(img), f"mask path {img} doesn't exist."
        self.transform_image = transform_image
        self.transform_mask = transform_mask
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        image = ImageOps.exif_transpose(image)  # avoid EXIF info in images
        mask = Image.open(self.mask_list[idx]).convert("L")
        mask = ImageOps.exif_transpose(mask)
        assert image.size == mask.size, f"image {image.size} and mask {mask.size} sizes are not same"
        image = self.transform_image(image) if self.transform_image is not None else image
        mask = self.transform_mask(mask) if self.transform_mask is not None else mask
        return image, mask


class ShaCaption(Dataset):
    def __init__(self, json_root, transform=None, split='Train', segmented=False, clean=False, captions_per_image=1, vocab=None):
        self.json_root = json_root
        self.captions_per_image = captions_per_image
        with open(self.json_root, 'r', encoding='gbk') as f:
            data = json.load(f)
        self.path_img, self.captions, self.multilabel = [], [], []
        for e in data['annotations']:
            if e['split'] == split:
                self.path_img.append(os.path.join(e['file_path'], e['file_name'] if not segmented else e['file_name'].replace('.jpg', '_seg.jpg')))
                for i, s in enumerate(e['sentences']):
                    if i < captions_per_image:
                        if clean:
                            self.captions.append(clean_caption(s['tokens']))
                        else:
                            self.captions.append(s['tokens'])
                multilabel = np.zeros(9, dtype=int)
                ltoi = {'平和质': 0, '气虚质': 1, '阳虚质': 2, '阴虚质': 3, '痰湿质': 4,
                        '湿热质': 5, '血瘀质': 6, '气郁质': 7, '特禀质': 8}
                for i, l in enumerate(e['labels']):
                    if float(l['score']) > 40 and l['constitution'] != '平和质':
                        multilabel[ltoi[l['constitution']]] = 1
                self.multilabel.append(multilabel)
        assert len(self.path_img) * captions_per_image == len(self.captions) == len(self.multilabel) * captions_per_image
        self.vocab = vocab
        self.transform = transform
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        image_path = self.path_img[idx // self.captions_per_image]
        image = Image.open(image_path).convert("RGB")
        caption = self.captions[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        embed_caption = [self.vocab.stoi["<SOS>"]]
        embed_caption += self.vocab.numericalize(caption)
        embed_caption.append(self.vocab.stoi["<EOS>"])
        
        multilabel = self.multilabel[idx // self.captions_per_image]
        
        return image, torch.tensor(embed_caption), multilabel


class Flickr8k(Dataset):
    def __init__(self, root, transforms=None, txt_name="Flickr_8k.trainImages.txt", freq_threshold=5, vocab=None):
        self.root = root  # './Flickr8k/Flickr8k_Dataset'
        self.img_root = os.path.join(self.root, 'Images')  # 图片文件夹路径
        self.token_root = os.path.join(self.root, 'Flickr8k.token.txt')  # token文件路径
        # self.token_root = os.path.join(self.root, 'token.txt')  # token文件路径
        self.txt_root = os.path.join(self.root, txt_name)  # txt文件路径
        
        raw_captions = {}
        raw_token_list = open(self.token_root, 'r', encoding='utf-8').read().strip().split('\n')
        self.captions_list = []
        for line in raw_token_list:
            line = line.split('\t')
            img_id, img_caption = line[0][:line[0].find('#')], line[1]
            self.captions_list.append(img_caption)
            if img_id not in raw_captions:
                raw_captions[img_id] = [img_caption]
            else:
                raw_captions[img_id].append(img_caption)
        
        txt_image_list = open(self.txt_root, 'r').read().strip().split('\n')  # 读取txt里面的文件名字
        self.image_captions_list = []  # 图像文件名与与之对应的caption
        for img_id in txt_image_list:
            for caption in raw_captions[img_id]:
                self.image_captions_list.append([img_id, caption])
        
        # 建立vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold=freq_threshold)
            self.captions_list = [e for s in self.captions_list for e in s.split(' ')]
            self.vocab.build_vocabulary(self.captions_list)
        
        # transform
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_captions_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_root, self.image_captions_list[idx][0])
        image_caption = self.image_captions_list[idx][1]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        
        image_caption = image_caption.split(' ')
        embed_caption = [self.vocab.stoi["<SOS>"]]
        embed_caption += self.vocab.numericalize(image_caption)
        embed_caption.append(self.vocab.stoi["<EOS>"])
        
        multilabel = np.zeros(9, dtype=int)
        return image, torch.tensor(embed_caption), multilabel
