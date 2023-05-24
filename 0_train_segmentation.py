from model.UNet import UNet
import torch
import os
from torchvision import transforms
from my_dataset import SegDataset
from torch.utils.data import DataLoader, random_split
from utils.collate_fn import MyCollate
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from utils.optim import ScheduledOptim
from utils.dice_score import dice_loss
from utils import ConfusionMatrix
from torchsummary import summary
from PIL import Image


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(seed)
        random.seed(seed)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
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
    
    data_root = args.dataset_path
    dataset = SegDataset(data_root, data_transform['image'], data_transform['mask'])
    print(f"[INFO] first tensors of images and mask in dataset\n"
          f"image: {dataset[0][0].shape} \n"
          f"mask: {dataset[0][1].shape} \n"
          f"len: {len(dataset)}")
    # split dataset, train/0.9, val/0.1
    
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print('dataset split!')
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])  # number of workers
    print('Using %g dataloader workers' % nw)
    
    loader_args = dict(num_workers=nw, pin_memory=True)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **loader_args)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, **loader_args)
    
    print(f"[INFO] Model Init")
    model = UNet(n_channels=3, n_classes=2).to(device)
    summary(model, input_size=(3, 224, 224))
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: %.2fM" % (total / 1e6))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    model.train()
    loss_train, loss_val = [], []
    best_F1 = 0.0
    print("<-Start Training!->")
    for epoch in range(args.epochs):
        print(f"[INFO] Epoch {epoch + 1} start training!")
        for step, (images, masks) in enumerate(train_data_loader):
            images = images.to(device=device, dtype=torch.float32)  # (N, C, H, W)
            masks = masks.to(device=device, dtype=torch.long).squeeze(1)  # (N, H, W)
            optimizer.zero_grad()
            
            masks_pred = model(images)  # (N, 2, H, W)
            # print(torch.sum(torch.argmax(masks_pred, dim=1)))
            if step == 1000:
                temp = masks_pred.argmax(dim=1)[0].cpu().numpy()
                out = np.zeros((temp.shape[-2], temp.shape[-1]), dtype=bool)
                out[temp == 1] = 1
                out = Image.fromarray(out)
                out.show()
            if model.n_classes == 1:
                loss = criterion(masks_pred.squeeze(1), masks.float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), masks.float(), multiclass=False)
            else:
                loss = criterion(masks_pred, masks)
                loss_dice = dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                loss += loss_dice
            
            loss_train.append(loss.item())
            print(f"Training: Epoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(train_data_loader)}], Dice Loss: {loss_dice:.4f}, Loss: {loss_train[-1]:.4f}")
            # backward
            loss.backward()
            optimizer.step()
        
        print(f"<-Start Validating!->")
        model.eval()
        batch_loss_val = []
        conf_mat = ConfusionMatrix(model.n_classes)
        for val_step, (images_val, masks_val) in enumerate(val_data_loader):
            images_val = images_val.to(device=device, dtype=torch.float32)  # (B, C, H, W)
            masks_val = masks_val.to(device=device, dtype=torch.long).squeeze(1)  # (B, H, W)
            masks_pred_val = model(images_val)
            if model.n_classes == 1:
                val_loss = criterion(masks_pred_val.squeeze(1), masks_val.float())
                val_loss += dice_loss(F.sigmoid(masks_pred_val.squeeze(1)), masks_val.float(), multiclass=False)
            else:
                val_loss = criterion(masks_pred_val, masks_val)
                val_loss_dice = dice_loss(
                    F.softmax(masks_pred_val, dim=1).float(),
                    F.one_hot(masks_val, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                val_loss += val_loss_dice
            conf_mat.update(masks_val.flatten(), masks_pred_val.argmax(1).flatten())
            batch_loss_val.append(val_loss.item())
            print(f"Validation: Epoch [{epoch + 1}/{args.epochs}], Step [{val_step + 1}/{len(val_data_loader)}], Dice Loss: {val_loss_dice:.4f}, Loss: {batch_loss_val[-1]:.4f}")
        mIoU, accuracy, precision, recall, F1 = conf_mat.compute()
        loss_val.append(np.mean(batch_loss_val))
        # save weights
        if best_F1 < F1:
            best_F1 = F1
            save_files = {
                'model': model.state_dict(),
                'args': args,
                'epoch': epoch + 1,
                'loss_train': loss_train,
                'val_loss': loss_val,
                'metric': [mIoU, accuracy, precision, recall, F1]
            }
            torch.save(save_files, args.save_path + "imageSeg-model-best.pth")
            print(f'epoch {epoch + 1} best.pth save! mIoU {mIoU:.4f}, accuracy {accuracy:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1 {F1:.4f}')


if __name__ == '__main__':
    import argparse
    # epoch 24 best.pth save! mIoU 0.9145, accuracy 0.9514, precision 0.9607, recall 0.9501, F1 0.9554
    parser = argparse.ArgumentParser()
    # BASIC SETTING
    parser.add_argument('--device', default='cuda:0', help='device')
    # parser.add_argument('--dataset_path', default='./Seg Dataset/DRIVE/training/', help='dataset path')
    parser.add_argument('--dataset_path', default='./Seg Dataset/', help='dataset path')
    # parser.add_argument('--dataset_path', default='./Sha Dataset', help='dataset path')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--save_path', default='./save/ImageSeg-UNet/', help='save path of model weight, log, et.al.')
    # TRAINING SETTING
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=25, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=3, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    
    args = parser.parse_args()
    print(f"args:{args}")
    main(args)
