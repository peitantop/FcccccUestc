from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import os
import numpy as np
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from collections import Counter
import sys
import json
from pathlib import Path

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


class EnhancedEyeDataset(Dataset):
    """
    增强版眼底图像数据集，支持多种预处理和数据增强策略
    """
    def __init__(self, 
                 data_dir, 
                 label_path, 
                 transform=None, 
                 preprocessing=None,
                 crop_size=224, 
                 is_test=False,
                 use_both_eyes=False,
                 label_smoothing=0.0,
                 return_path=False):
        """
        初始化数据集
        
        参数:
            data_dir (str): 图像数据目录
            label_path (str): 标签文件路径
            transform (callable, optional): 图像变换函数
            preprocessing (callable, optional): 预处理函数
            crop_size (int): 裁剪尺寸
            is_test (bool): 是否为测试模式
            use_both_eyes (bool): 是否同时使用左右眼图像
            label_smoothing (float): 标签平滑系数
            return_path (bool): 是否返回图像路径
        """
        self.data_dir = data_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.crop_size = crop_size
        self.is_test = is_test
        self.use_both_eyes = use_both_eyes
        self.label_smoothing = label_smoothing
        self.return_path = return_path
        
        # 读取数据
        self.df = pd.read_csv(label_path, header=0, encoding='utf-8')
        
        # 提取患者ID (假设文件名格式为: ID_left.jpg 或 ID_right.jpg)
        self.df['patient_id'] = self.df['Left-Fundus'].apply(lambda x: x.split('_')[0])
        
        # 存储所有患者ID
        self.patient_ids = self.df['patient_id'].unique()
        
        # 获取标签列名
        self.label_columns = self.df.columns[5:13].tolist()
        
        # 存储标签
        self.labels = self.df.iloc[:, 5:13].values.astype(int)
        
        # 分析标签分布
        self.analyze_label_distribution()

    def __len__(self):
        return len(self.patient_ids) if self.is_test else len(self.df)
    
    def analyze_label_distribution(self):
        """
        分析标签分布情况
        """
        label_counts = {}
        for i, col in enumerate(self.label_columns):
            count = self.df[col].sum()
            label_counts[col] = count
            print(f"类别 {col}: {count} 样本 ({count/len(self.df)*100:.2f}%)")
        
        # 计算多标签组合
        label_combinations = self.df.iloc[:, 5:13].apply(lambda x: ''.join(x.astype(str)), axis=1)
        combination_counts = Counter(label_combinations)
        print(f"\n共有 {len(combination_counts)} 种不同的标签组合")
        print("前5种最常见的标签组合:")
        for combo, count in combination_counts.most_common(5):
            print(f"组合 {combo}: {count} 样本 ({count/len(self.df)*100:.2f}%)")
        
        self.label_counts = label_counts
        self.combination_counts = combination_counts
    
    def get_class_weights(self):
        """
        计算类别权重，用于处理类别不平衡
        """
        weights = []
        for i, col in enumerate(self.label_columns):
            pos_count = self.df[col].sum()
            neg_count = len(self.df) - pos_count
            # 避免除零错误
            if pos_count == 0:
                weights.append(1.0)
            else:
                # 计算权重: 负样本数/正样本数
                weights.append(neg_count / pos_count)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _crop_circle(self, img):
        """
        将正方形图片裁剪为圆形，保留眼底区域
        """
        mask = Image.new('L', (self.crop_size, self.crop_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, self.crop_size, self.crop_size), fill=255)
        result = Image.new('RGB', (self.crop_size, self.crop_size), (0, 0, 0))
        result.paste(img, (0, 0), mask=mask)
        return result
    
    def _apply_label_smoothing(self, label):
        """
        应用标签平滑
        """
        if self.label_smoothing <= 0:
            return label
        
        # 对于正样本，减小标签值；对于负样本，增加标签值
        smoothed_label = torch.where(
            label > 0.5,
            label - self.label_smoothing,
            label + self.label_smoothing
        )
        return smoothed_label

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
            left_image = self._crop_circle(left_image)
            right_image = self._crop_circle(right_image)
            
            # 对右眼图像进行左右翻转，使其与左眼方向一致
            right_image = right_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 应用预处理
            if self.preprocessing:
                left_image = self.preprocessing(left_image)
                right_image = self.preprocessing(right_image)
            
            # 应用变换
            if self.transform:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)
            
            # 获取标签
            label = torch.tensor(patient_data.iloc[5:13].values.astype(float), dtype=torch.float32)
            
            # 应用标签平滑
            label = self._apply_label_smoothing(label)
            
            if self.return_path:
                return (left_image, right_image), label, (left_img_name, right_img_name)
            else:
                return (left_image, right_image), label
        else:
            # 训练/验证模式
            row = self.df.iloc[idx]
            
            # 确定是左眼还是右眼
            is_left_eye = '_left' in row['Left-Fundus'].lower()
            left_img_name = row['Left-Fundus']
            right_img_name = row['Right-Fundus']
            
            if self.use_both_eyes:
                # 同时使用左右眼
                left_img_path = os.path.join(self.data_dir, left_img_name)
                right_img_path = os.path.join(self.data_dir, right_img_name)
                
                # 加载并处理图片
                left_image = default_loader(left_img_path)
                right_image = default_loader(right_img_path)
                
                # 统一缩放为正方形
                left_image = left_image.resize((self.crop_size, self.crop_size))
                right_image = right_image.resize((self.crop_size, self.crop_size))
                
                # 圆形裁剪
                left_image = self._crop_circle(left_image)
                right_image = self._crop_circle(right_image)
                
                # 对右眼图像进行左右翻转
                right_image = right_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 应用预处理
                if self.preprocessing:
                    left_image = self.preprocessing(left_image)
                    right_image = self.preprocessing(right_image)
                
                # 应用变换
                if self.transform:
                    left_image = self.transform(left_image)
                    right_image = self.transform(right_image)
                
                # 获取标签
                label = torch.tensor(row.iloc[5:13].values.astype(float), dtype=torch.float32)
                
                # 应用标签平滑
                label = self._apply_label_smoothing(label)
                
                if self.return_path:
                    return (left_image, right_image), label, (left_img_name, right_img_name)
                else:
                    return (left_image, right_image), label
            else:
                # 使用单眼
                img_name = left_img_name if is_left_eye else right_img_name
                img_path = os.path.join(self.data_dir, img_name)
                
                # 加载并处理图片
                image = default_loader(img_path)
                
                # 统一缩放为正方形
                image = image.resize((self.crop_size, self.crop_size))
                
                # 圆形裁剪
                image = self._crop_circle(image)
                
                # 对右眼图像进行左右翻转
                if not is_left_eye:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 应用预处理
                if self.preprocessing:
                    image = self.preprocessing(image)
                
                # 应用变换
                if self.transform:
                    image = self.transform(image)
                
                # 获取标签
                label = torch.tensor(row.iloc[5:13].values.astype(float), dtype=torch.float32)
                
                # 应用标签平滑
                label = self._apply_label_smoothing(label)
                
                if self.return_path:
                    return image, label, img_name
                else:
                    return image, label


def get_transforms(mode='train', size=224, use_albumentations=True):
    """
    获取图像变换函数
    
    参数:
        mode (str): 'train', 'valid' 或 'test'
        size (int): 图像大小
        use_albumentations (bool): 是否使用albumentations库
    
    返回:
        transform (callable): 变换函数
    """
    if use_albumentations:
        if mode == 'train':
            return A.Compose([
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=0.5),
                    A.ElasticTransform(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:  # valid or test
            return A.Compose([
                A.Resize(height=size, width=size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
