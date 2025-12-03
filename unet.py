# -*- coding: utf-8 -*-

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
#import rasterio
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from monai.networks.nets import  UNet
from monai.networks.layers import Norm
from  monai.networks import  normal_init
import time

# ============================== 配置 ==============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

EPOCHS = 10
BATCH_SIZE = 64
LR = 2e-4
# =================================================================

class UNet_monai(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        self.unet = UNet(spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            norm=Norm.BATCH)
        self.unet.apply(normal_init)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        output = self.unet(x)
        return output

class PointDataset(Dataset):
    def __init__(self, images_path: list, label_path: list, class_map:dict, transform=None, use_fraction=1.0, is_train=True):
        self.images_path = images_path
        self.label_path = label_path
        self.transform = transform
        self.use_fraction = use_fraction  # 使用多少比例的标注点，如0.33表示1/3
        self.is_train = is_train  # 是否训练模式
        self.class_map = class_map
    
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label_np = np.loadtxt(self.label_path[item], delimiter=',').astype(np.uint8)
        
        # 如果是训练模式且use_fraction<1，则随机采样部分标注点
        if self.is_train and self.use_fraction < 1.0:
            # 找出所有非零标注点的坐标
            y_indices, x_indices = np.where(label_np > 0)
            num_points = len(y_indices)
            
            # 计算需要采样的点数
            sample_size = max(1, int(num_points * self.use_fraction))
            
            # 随机选择部分点
            if num_points > 0:
                selected_indices = np.random.choice(num_points, sample_size, replace=False)
                sampled_label = np.zeros_like(label_np)
                sampled_label[y_indices[selected_indices], x_indices[selected_indices]] = \
                    label_np[y_indices[selected_indices], x_indices[selected_indices]]
                label_np = sampled_label

        # 其余处理保持不变
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        target_size = (512, 512)
        img = img.resize(target_size, Image.BILINEAR)
        label_np = cv2.resize(label_np, target_size, interpolation=cv2.INTER_NEAREST)
        
        if self.transform is not None:
            img = self.transform(img)
        
        mapped_label = np.zeros_like(label_np, dtype=np.uint8)
        for k, v in self.class_map.items():
            mapped_label[label_np == k] = v
        
        label_tensor = torch.from_numpy(mapped_label).long()
        
        return {'img': img, 'seg': label_tensor}    

'''def get_img(img_path):
    print("正在读取影像...")    
    # 读取影像（自动处理 16bit → 8bit）
    img = cv2.imread(HR_IMAGE, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:          # 16bit 转 8bit
        img = (img / 257).astype(np.uint8)   # 65535 / 257 ≈ 255
    if len(img.shape) == 2:             # 单通道变三通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0   # 归一化
    h, w = img.shape[:2]

    # 读取 10m 标签（单通道）
    lr = cv2.imread(LR_LABEL, cv2.IMREAD_GRAYSCALE)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)

    lr_mapped = np.zeros_like(lr_up, dtype=np.uint8)
    for k, v in CLASS_MAP.items():
        lr_mapped[lr_up == k] = v

    mask = lr_mapped > 0
    coords = np.stack(np.nonzero(mask), axis=1)        # [N, 2]  y,x
    labels = (lr_mapped[mask] - 1).astype(np.int64)    # 0~4
    return img, coords, labels'''


def train(train_image, label_image, class_map, model_name, Bach_size=2, use_fraction=0.3):
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(4)

    #train_image_path = ['data/data1/pic_resize.png', 'data/data2/pic_resize.png', 'data/data3/pic_resize.png', 'data/data4/pic_resize.png']
    #train_label_path = ['data/data1/prompt.png', 'data/data2/prompt.png', 'data/data3/prompt.png', 'data/data4/prompt.png']
    #train_label_path = ['data/result1/accurate.csv', 'data/result2/accurate.csv', 'data/result3/accurate.csv', 'data/result4/accurate.csv']
    train_image_path = [train_image]
    train_label_path = [label_image]
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                            std=[0.229, 0.224, 0.225])   # Standard ImageNet std
    ])
    #dataset = PointDataset(images_path=train_image_path,
    #                          label_path=train_label_path,
    #                          transform=train_transform)

    dataset = PointDataset(
        images_path=train_image_path,
        label_path=train_label_path,
        class_map=class_map,
        transform=train_transform,
        use_fraction=use_fraction,
        is_train=True
    )

    loader = DataLoader(dataset, batch_size=Bach_size, shuffle=False, num_workers=4)

    model = UNet_monai(n_classes=Bach_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=255)

    print("开始训练...")
    train_start = time.time()
    model.train()
    
    scaler = torch.cuda.amp.GradScaler()

    print('begin to epoch')

    for epoch in range(EPOCHS):
        for batch in loader:
            x = batch['img'].to(DEVICE)
            y = batch['seg'].to(DEVICE)  # Should be [batch_size, 1, H, W]

            print(f"Input shape: {x.shape}")  # Should be [batch_size, 3, H, W]
            print(f"Target shape: {y.shape}")  # Should be [batch_size, H, W]
            
            # Ensure target is 3D [batch_size, H, W] with class indices
            if y.dim() == 4:
                y = y.squeeze(1)  # Remove channel dim if present
            y = y.long()  # Convert to LongTensor
            
            with torch.cuda.amp.autocast():
                p = model(x)  # Should be [batch_size, num_classes, H, W]
                loss = crit(p, y)
            print(f"Model output shape: {p.shape}")  # Should be [batch_size, num_classes, H, W]
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            

    print(f"【U-Net 显卡】训练完成！耗时: {time.time()-train_start:.2f} 秒")

    torch.save(model, model_name)
  
def predict(model_file, HR_IMAGE, OUTPUT_DIR, n_classes=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = torch.load(model_file, map_location=DEVICE)
    model.eval()

    print("正在读取影像...")    
    # 读取影像（自动处理 16bit → 8bit）
    img = cv2.imread(HR_IMAGE, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:          # 16bit 转 8bit
        img = (img / 257).astype(np.uint8)   # 65535 / 257 ≈ 255
    if len(img.shape) == 2:             # 单通道变三通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0   # 归一化

    img = img.transpose(2, 0, 1)  # [3,H,W]
    
    # Calculate required padding
    h, w = img.shape[1], img.shape[2]
    target_h = ((h + 15) // 16) * 16
    target_w = ((w + 15) // 16) * 16
    pad_h = target_h - h
    pad_w = target_w - w
    
    # Pad the image
    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    with torch.no_grad():
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)
        pred = model(img_tensor)
        
        # If model returns a tuple, take first element
        if isinstance(pred, tuple):
            pred = pred[0]
        
        # Remove padding from prediction
        pred = pred[:, :, :h, :w]
        
        # Get the class with highest probability for each pixel
        pred_map = torch.argmax(pred, dim=1)  # Shape: [batch_size, H, W]
        pred_map = pred_map.squeeze().cpu().numpy().astype(np.uint8)
        
        # For binary classification, we only care about class 1 (foreground)
        if n_classes == 2:
            pred_map = (pred_map == 1).astype(np.uint8)  # Convert to 0 and 1
            colors = np.array([[0, 0, 0], [0, 255, 0]], dtype=np.uint8)  # Black and Green
            output_name = f"unet_binary_result{output_suffix}.png"
            csv_name = f"unet_binary_labels{output_suffix}.csv"
        else:
            pred_map += 1  # Shift class indices to start from 1
            colors = np.array([
                [0, 0, 0],      # Background
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 0, 0],    # Red
                [255, 255, 0],  # Yellow
                [139, 69, 19]   # Brown
            ], dtype=np.uint8)
            output_name = "unet_result.png"
            csv_name = "unet_labels.csv"
    
    # Create colored output
    if n_classes == 2:
        colored_pred = colors[pred_map]
    else:
        colored_pred = colors[pred_map - 1]  # Shift back to 0-based index for colors
    
    # Save the results
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, output_name),
        cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
    )
    
    # Save as CSV
    pd.DataFrame(pred_map).to_csv(
        os.path.join(OUTPUT_DIR, csv_name),
        index=False,
        header=False
    )

    mode = "二分类" if n_classes == 2 else f"{n_classes}类"
    print(f"U-Net {mode}完成！结果已保存到 {os.path.join(OUTPUT_DIR, output_name)}")

def predict_2(model_file, HR_IMAGE, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = torch.load(model_file, map_location=DEVICE)
    model.eval()

    print("正在读取影像...")    
    # 读取影像（自动处理 16bit → 8bit）
    img = cv2.imread(HR_IMAGE, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint16:          # 16bit 转 8bit
        img = (img / 257).astype(np.uint8)   # 65535 / 257 ≈ 255
    if len(img.shape) == 2:             # 单通道变三通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0   # 归一化

    img = img.transpose(2, 0, 1)  # [3,H,W]
    
    # Calculate required padding
    h, w = img.shape[1], img.shape[2]
    target_h = ((h + 15) // 16) * 16
    target_w = ((w + 15) // 16) * 16
    pad_h = target_h - h
    pad_w = target_w - w
    
    # Pad the image
    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    with torch.no_grad():
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)
        pred = model(img_tensor)
        
        # If model returns a tuple, take first element
        if isinstance(pred, tuple):
            pred = pred[0]
        
        # Remove padding from prediction
        pred = pred[:, :, :h, :w]
        
        # Get the class with highest probability for each pixel
        pred_map = torch.argmax(pred, dim=1)  # Shape: [batch_size, H, W]
        pred_map = pred_map.squeeze().cpu().numpy().astype(np.uint8)  # Shape: [H, W]
        
        # For binary classification, we only care about class 1 (assuming it's the foreground class)
        binary_mask = (pred_map == 1).astype(np.uint8)  # Convert to 0 and 1

    # For binary segmentation, we only need two colors (background and foreground)
    colors = np.array([[0, 0, 0], [0, 255, 0]], dtype=np.uint8)
    
    # Create colored output
    colored_pred = colors[binary_mask]
    
    # Save the results
    cv2.imwrite(os.path.join(OUTPUT_DIR, "unet_binary_result.png"), 
               cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
    
    # Save as CSV - ensure it's 2D
    pd.DataFrame(binary_mask).to_csv(
        os.path.join(OUTPUT_DIR, "unet_binary_labels.csv"), 
        index=False, 
        header=False
    )

    print("U-Net 二分类完成！结果已保存到", OUTPUT_DIR)

if __name__ == '__main__':
    #CLASS_MAP = {20:0, 30:1, 60:2, 70:3, 90:4}
    #CLASS_MAP = {20:0, 30:1}
    CLASS_MAP = {20:0, 60:1, 70:2, 90:3}
    #train("data/data4/pic_resize.png","data/result4/accurate.csv",CLASS_MAP,'./unet-4-33.pt',4,1)
    predict("./unet-4-33.pt", "data/data4/pic_resize.png", "data/result4/unet-4-33",4)