import os
import shutil
from model.UNet import UNet
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image  # Load img
from model.NIC import EncoderCNN, DecoderRNN
from my_dataset import Flickr8k
import glob
from torch.utils.data import DataLoader
from utils.collate_fn import MyCollate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.clean_captions import clean_caption
import numpy as np
from pylab import mpl
import torch.nn.functional as F
from utils.image_process import remove_small_object, seg

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def predict_img(model,
                full_img,
                image,
                out_threshold=0.1):
    model.eval()
    with torch.no_grad():
        output = model(image)
        print(output.shape)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if model.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    print(mask.shape)
    exit()
    return mask[0].long().squeeze().numpy()

def main(opts):
    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
    data_transform = {
        "image": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        "mask": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    }
    
    weights_path = "./save/ImageSeg-UNet/imageSeg-model-25.pth"
    weights_dict = torch.load(weights_path, map_location='cpu')
    args = weights_dict['args']
    
    model = UNet(n_channels=3, n_classes=2).cpu()
    model.eval()
    model.load_state_dict(weights_dict["model"])


    # paths = glob.glob('./Seg Dataset/DRIVE/training/JPEGImages/' + '*.tif')
    # paths = glob.glob('./Seg Dataset/JPEGImages/' + '*痧象*.jpg')
    paths = glob.glob('./Dataset - ALL/*/*/' + '*.jpg')
    for path in paths:
        if '_seg' in path:
            continue
        image_pil = Image.open(path).convert("RGB")
        image = data_transform['image'](image_pil).unsqueeze(0)
        image = image.to(dtype=torch.float32)
        pred = predict_img(model, Image.open(path).convert("RGB"), image)
        mask = np.zeros((pred.shape[-2], pred.shape[-1]), dtype=bool)
        for i, v in enumerate({0: 0, 1: 1}):
            mask[pred == i] = v
        
        from skimage import morphology
        mask = mask.astype(bool)
        mask = morphology.remove_small_objects(mask, 5e4, connectivity=2)
        mask = Image.fromarray(mask)
        mask = seg(mask)
        mask.save(os.path.split(path)[0] + '/' + os.path.split(path)[1].replace('.jpg', '_mask.jpg'))

        # mask = Image.fromarray(mask)
        # temp = Image.new('RGB', image_pil.size, (0, 0, 0))
        # image_rgb = Image.composite(image_pil, temp, mask)
        # image_res = seg(remove_small_object(image_rgb.convert('L'), image_rgb))
        # image_res.save(os.path.split(path)[0] + '/' + os.path.split(path)[1].replace('.jpg', '_seg.jpg'))
        print(f"{path} processed!")



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    # BASIC SETTING
    parser.add_argument('--device', default='cuda:0', help='device')
    # parser.add_argument('--dataset_path', default='./Seg Dataset/DRIVE/training/', help='dataset path')
    parser.add_argument('--dataset_path', default='./Seg Dataset/', help='dataset path')
 
    opt = parser.parse_args()
    print(f"opt:{opt}")
    
    main(opt)
