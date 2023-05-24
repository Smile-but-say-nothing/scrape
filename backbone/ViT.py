import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViT(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=9)
        self.fc = nn.Linear(768, 2048)
        
    def forward(self, x):
        features = self.vit.forward_features(x)
        features = self.fc(features)
        
        fc = self.vit(x)
        
        return features, fc
    

if __name__ == '__main__':
    net = ViT('vit_base_patch32_224')
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True
    a = [p for p in net.parameters() if p.requires_grad]
    print(a[0].shape, a[1].shape)
    print(len(a))
    x = torch.randn(6, 3, 224, 224)
    output = net(x)
    print(output[0].shape, output[1].shape)
        