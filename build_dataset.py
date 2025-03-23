from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms, datasets
import pandas as pd
from PIL import Image, ImageDraw
import os
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import sys
import re
import cv2
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')




data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'
# Excel 文件路径
generator = torch.Generator().manual_seed(42)

# 将 DataFrame 保存为 CSV 文件
df = pd.read_csv(label_dir, header=0, encoding='utf-8')


class EyeDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, crop_size=224, is_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size
        self.is_test = is_test
        
        # 读取数据
        self.df = pd.read_csv(label_dir, header=0, encoding='utf-8')
        
        # 提取患者ID (假设文件名格式为: ID_left.jpg 或 ID_right.jpg)
        self.df['patient_id'] = self.df['Left-Fundus'].apply(lambda x: x.split('_')[0])
        
        # 存储所有患者ID
        self.patient_ids = self.df['patient_id'].unique()
        
        # 存储标签
        self.labels = self.df.iloc[:, 5:13].values.astype(int)

    def __len__(self):
        return len(self.patient_ids) if self.is_test else len(self.df)

    def _find_circle_crop_params(self, img):
        """找到最佳的圆形裁剪参数"""
        # 转换为灰度图
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        # 使用Otsu's二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 找到非零区域的边界
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)
        
        # 计算圆心和半径
        center_x = x + w // 2
        center_y = y + h // 2
        radius = min(w, h) // 2
        
        return center_x, center_y, radius

    def _smart_crop_circle(self, img):
        """智能圆形裁剪"""
        # 找到最佳裁剪参数
        center_x, center_y, radius = self._find_circle_crop_params(img)
        
        # 创建圆形掩码
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 绘制圆形
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
        draw.ellipse(bbox, fill=255)
        
        # 应用掩码并裁剪
        result = Image.new('RGB', (radius * 2, radius * 2), (0, 0, 0))
        cropped = img.crop(bbox)
        result.paste(cropped, (0, 0), mask.crop(bbox))
        
        # 调整到目标大小
        result = result.resize((self.crop_size, self.crop_size))
        return result

    def __getitem__(self, idx):
        if self.is_test:
            # 测试模式：返回患者的左右眼图像和标签
            patient_id = self.patient_ids[idx]
            patient_data = self.df[self.df['patient_id'] == patient_id].iloc[0]
            
            left_img_name = patient_data['Left-Fundus']
            right_img_name = patient_data['Right-Fundus']
            
            left_img_path = os.path.join(self.data_dir, left_img_name)
            right_img_path = os.path.join(self.data_dir, right_img_name)
            
            # 加载并处理图片
            left_image = default_loader(left_img_path)
            right_image = default_loader(right_img_path)
            
            # 统一缩放为正方形
            left_image = left_image.resize((self.crop_size, self.crop_size))
            right_image = right_image.resize((self.crop_size, self.crop_size))
            
            # 圆形裁剪
            left_image = self._smart_crop_circle(left_image)
            right_image = self._smart_crop_circle(right_image)
            
            # 对右眼图像进行左右翻转
            right_image = right_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 应用变换
            if self.transform:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)
            
            label = torch.tensor(patient_data.iloc[5:13].values.astype(int), dtype=torch.float32)
            
            return (left_image, right_image), label, (left_img_name, right_img_name)
        else:
            # 训练/验证模式：返回单个眼睛图像和标签
            row = self.df.iloc[idx]
            
            # 确定是左眼还是右眼
            is_left_eye = '_left' in row['Left-Fundus'].lower()
            img_name = row['Left-Fundus'] if is_left_eye else row['Right-Fundus']
            
            img_path = os.path.join(self.data_dir, img_name)
            
            # 加载并处理图片
            image = default_loader(img_path)
            
            # 统一缩放为正方形
            image = image.resize((self.crop_size, self.crop_size))
            
            # 圆形裁剪
            image = self._smart_crop_circle(image)
            
            # 对右眼图像进行左右翻转
            if not is_left_eye:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor(row.iloc[5:13].values.astype(int), dtype=torch.float32)
            
            return image, label


def split_dataset_by_patient(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    按患者ID划分数据集，确保同一患者的左右眼在同一数据集中
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例总和必须为1"
    
    # 提取患者ID
    df['patient_id'] = df['Left-Fundus'].apply(lambda x: x.split('_')[0])
    patient_ids = df['patient_id'].unique()
    
    # 随机打乱患者ID
    np.random.seed(random_seed)
    np.random.shuffle(patient_ids)
    
    # 计算每个集合的患者数量
    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # 划分患者ID
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train+n_val]
    test_ids = patient_ids[n_train+n_val:]
    
    # 根据患者ID筛选数据
    train_df = df[df['patient_id'].isin(train_ids)]
    val_df = df[df['patient_id'].isin(val_ids)]
    test_df = df[df['patient_id'].isin(test_ids)]
    
    print(f"训练集: {len(train_df)} 样本, {len(train_ids)} 患者")
    print(f"验证集: {len(val_df)} 样本, {len(val_ids)} 患者")
    print(f"测试集: {len(test_df)} 样本, {len(test_ids)} 患者")
    
    return train_df, val_df, test_df


# 图像变换
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        # 将PIL图像转换为OpenCV格式
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # 对L通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img[:,:,0] = clahe.apply(img[:,:,0])
        
        # 转换回RGB
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

# 修改训练数据增强
train_transforms = transforms.Compose([
    transforms.Resize(224),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8,8)),  # 添加CLAHE
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def create_dataloaders(data_dir, label_dir, batch_size=32, num_workers=0):
    """
    创建训练、验证和测试数据加载器
    """
    # 读取数据
    df = pd.read_csv(label_dir, header=0, encoding='utf-8')
    
    # 按患者ID划分数据集
    train_df, val_df, test_df = split_dataset_by_patient(df)
    
    # 保存划分后的数据集
    train_df.to_csv('D:/Fc25_07/FcccccUestc/train_data.csv', index=False)
    val_df.to_csv('D:/Fc25_07/FcccccUestc/val_data.csv', index=False)
    test_df.to_csv('D:/Fc25_07/FcccccUestc/test_data.csv', index=False)
    
    # 创建数据集
    train_dataset = EyeDataset(data_dir, 'D:/Fc25_07/FcccccUestc/train_data.csv', transform=train_transforms)
    val_dataset = EyeDataset(data_dir, 'D:/Fc25_07/FcccccUestc/val_data.csv', transform=valid_transforms)
    test_dataset = EyeDataset(data_dir, 'D:/Fc25_07/FcccccUestc/test_data.csv', transform=test_transforms, is_test=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def denormalize(tensor, mean, std):
    """反归一化张量"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# 可视化函数

def show_images_and_labels(dataloader, num_images=5):
    # 获取一批数据
    batch = next(iter(dataloader))
    
    # 反归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if len(batch) == 3:  # 测试模式
        (left_images, right_images), labels, _ = batch
        
        # 确保我们不尝试显示超过批次大小的图像
        actual_num_images = min(num_images, left_images.size(0))
        
        left_images = denormalize(left_images[:actual_num_images], mean, std)
        right_images = denormalize(right_images[:actual_num_images], mean, std)
        
        # 创建画布
        fig, axes = plt.subplots(nrows=2, ncols=actual_num_images, figsize=(15, 8))
        
        # 处理单张图片的情况
        if actual_num_images == 1:
            # 转换为 PIL 图像
            left_img = left_images[0].permute(1, 2, 0)
            right_img = right_images[0].permute(1, 2, 0)
            left_img = np.clip(left_img.numpy(), 0, 1)
            right_img = np.clip(right_img.numpy(), 0, 1)
            
            # 显示图像
            axes[0].imshow(left_img)
            axes[0].set_title("left 1")
            axes[0].axis('off')
            
            axes[1].imshow(right_img)
            axes[1].set_title("right 1")
            axes[1].axis('off')
        else:
            for i in range(actual_num_images):
                # 转换为 PIL 图像
                left_img = left_images[i].permute(1, 2, 0)
                right_img = right_images[i].permute(1, 2, 0)
                left_img = np.clip(left_img.numpy(), 0, 1)
                right_img = np.clip(right_img.numpy(), 0, 1)
                
                # 显示图像
                axes[0, i].imshow(left_img)
                axes[0, i].set_title(f"left {i+1}")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(right_img)
                axes[1, i].set_title(f"right {i+1}")
                axes[1, i].axis('off')
        
        plt.suptitle(f"label: {labels[0].numpy()}")
    else:  # 训练/验证模式
        images, labels = batch
        
        # 确保我们不尝试显示超过批次大小的图像
        actual_num_images = min(num_images, images.size(0))
        
        images = denormalize(images[:actual_num_images], mean, std)
        
        # 创建画布
        fig, axes = plt.subplots(nrows=1, ncols=actual_num_images, figsize=(15, 5))
        
        # 处理单张图片的情况
        if actual_num_images == 1:
            # 转换为 PIL 图像
            img = images[0].permute(1, 2, 0)
            img = np.clip(img.numpy(), 0, 1)
            
            # 显示图像
            axes.imshow(img)
            axes.set_title(f"label: {labels[0].numpy()}")
            axes.axis('off')
        else:
            for i in range(actual_num_images):
                # 转换为 PIL 图像
                img = images[i].permute(1, 2, 0)
                img = np.clip(img.numpy(), 0, 1)
                
                # 显示图像
                axes[i].imshow(img)
                axes[i].set_title(f"label: {labels[i].numpy()}")
                axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 创建数据加载器
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        data_dir=data_dir,
        label_dir=label_dir,
        batch_size=32
    )
    
    # 查看训练集样本
    print("训练集样本:")
    show_images_and_labels(train_loader, num_images=5)
    
    # 查看测试集样本
    print("测试集样本:")
    show_images_and_labels(test_loader, num_images=3)