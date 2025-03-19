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
from skmultilearn.problem_transform import LabelPowerset
from collections import defaultdict
import random
import math
from matplotlib.font_manager import FontProperties

# 在创建图表之前添加以下代码来设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')




data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'
# Excel 文件路径
generator = torch.Generator().manual_seed(42)

# 将 DataFrame 保存为 CSV 文件
df = pd.read_csv(label_dir, header=0, encoding='utf-8')


class EyeDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, crop_size=384, is_test=False):
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

    def _crop_circle(self, img):
        """将正方形图片裁剪为圆形"""
        mask = Image.new('L', (self.crop_size, self.crop_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, self.crop_size, self.crop_size), fill=255)
        result = Image.new('RGB', (self.crop_size, self.crop_size), (0, 0, 0))
        result.paste(img, (0, 0), mask=mask)
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
            left_image = self._crop_circle(left_image)
            right_image = self._crop_circle(right_image)
            
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
            image = self._crop_circle(image)
            
            # 对右眼图像进行左右翻转
            if not is_left_eye:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor(row.iloc[5:13].values.astype(int), dtype=torch.float32)
            
            return image, label


def split_dataset_by_patient(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
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
train_transforms = transforms.Compose([
    transforms.Resize(224),
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

def distribute_remainder(r, r_dist, idx):
    """分配剩余的样本数量"""
    p = len(r_dist) - idx + 1
    value = r // p
    curr_rem = r % p

    r_dist[idx:] = np.add(r_dist[idx:], value)
    
    if curr_rem > 0:
        start = len(r_dist) - curr_rem
        r_dist[start:] = np.add(r_dist[start:], 1)


def LP_ROS(y, p):
    """
    标签幂集随机过采样方法
    
    参数:
    y: 标签矩阵，形状为 (n_samples, n_labels)
    p: 过采样百分比
    
    返回:
    需要克隆的样本索引列表
    """
    samples_to_clone = int(y.shape[0] * p / 100)

    lp = LabelPowerset()
    labelsets = np.array(lp.transform(y))
    label_set_bags = defaultdict(list)

    for idx, label in enumerate(labelsets):
        label_set_bags[label].append(idx)

    mean_size = 0
    for label, samples in label_set_bags.items():
        mean_size += len(samples)
    
    # 向上取整
    mean_size = math.ceil(mean_size / len(label_set_bags))

    minority_bag = []
    for label, samples in label_set_bags.items():
        if len(samples) < mean_size:
            minority_bag.append(label)

    if len(minority_bag) == 0:
        print('没有低于平均大小的标签组合。平均大小: ', mean_size)
        return []

    mean_increase = samples_to_clone // len(minority_bag)

    def custom_sort(label):
        return len(label_set_bags[label])

    minority_bag.sort(reverse=True, key=custom_sort)
    acc_remainders = np.zeros(len(minority_bag), dtype=np.int32)
    clone_samples = []

    for idx, label in enumerate(minority_bag):
        increase_bag = min(mean_size - len(label_set_bags[label]), mean_increase)

        remainder = mean_increase - increase_bag

        if remainder == 0:
            extra_increase = min(mean_size - len(label_set_bags[label]) - increase_bag, acc_remainders[idx])
            increase_bag += extra_increase
            remainder = acc_remainders[idx] - extra_increase

        distribute_remainder(remainder, acc_remainders, idx + 1)

        for i in range(increase_bag):
            x = random.randint(0, len(label_set_bags[label]) - 1)
            clone_samples.append(label_set_bags[label][x])

    return clone_samples


def balance_dataset(dataset, oversampling_percentage=50):
    """
    平衡数据集中的类别分布
    
    参数:
    dataset: EyeDataset实例
    oversampling_percentage: 过采样百分比
    
    返回:
    平衡后的数据集
    """
    # 获取所有标签
    all_labels = np.array([label.numpy() for _, label in dataset])
    
    # 使用LP_ROS方法获取需要克隆的样本索引
    clone_indices = LP_ROS(all_labels, oversampling_percentage)
    
    if not clone_indices:
        print("数据集已经平衡，无需过采样")
        return dataset
    
    print(f"将克隆 {len(clone_indices)} 个样本以平衡数据集")
    
    # 创建一个新的数据集类，继承自原始数据集类
    class BalancedEyeDataset(Dataset):
        def __init__(self, original_dataset, clone_indices):
            self.original_dataset = original_dataset
            self.clone_indices = clone_indices
            
        def __len__(self):
            return len(self.original_dataset) + len(self.clone_indices)
            
        def __getitem__(self, idx):
            if idx < len(self.original_dataset):
                return self.original_dataset[idx]
            else:
                # 获取克隆样本的原始索引
                original_idx = self.clone_indices[idx - len(self.original_dataset)]
                return self.original_dataset[original_idx]
    
    return BalancedEyeDataset(dataset, clone_indices)



def create_dataloaders(data_dir, label_dir, batch_size=32, num_workers=0, balance_train=True):
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
    
     # 如果需要平衡训练数据
    if balance_train:
        train_dataset = balance_dataset(train_dataset)

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


def analyze_label_distribution(dataset, title="标签分布"):
    """
    分析并打印数据集中各标签的分布情况
    
    参数:
    dataset: 数据集实例
    title: 分析标题
    """
    # 获取所有标签
    if hasattr(dataset, 'original_dataset'):
        # 对于平衡后的数据集，获取所有样本（包括原始和克隆的）
        all_samples = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_samples.append(label.numpy())
        labels = np.array(all_samples)
    else:
        # 对于原始数据集
        labels = np.array([label.numpy() for _, label in dataset])
    
    # 打印标签分布
    print(f"\n{title}:")
    for i in range(labels.shape[1]):
        positive_count = np.sum(labels[:, i])
        print(f"标签 {i+1}: 正例 {positive_count}, 负例 {len(labels) - positive_count}, 正例比例 {positive_count/len(labels):.2%}")
    
    # 绘制标签分布柱状图
    label_names = ["标签"+str(i+1) for i in range(labels.shape[1])]
    positive_ratios = [np.sum(labels[:, i])/len(labels) for i in range(labels.shape[1])]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(label_names, positive_ratios, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel('正例比例')
    plt.title(title)

        # 在柱状图上添加具体数值
    for bar, ratio in zip(bars, positive_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ratio:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 创建数据加载器
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        data_dir=data_dir,
        label_dir=label_dir,
        batch_size=32,
        balance_train=True  # 启用数据平衡
    )
    
    # 分析原始训练集标签分布
    if hasattr(train_dataset, 'original_dataset'):
        analyze_label_distribution(train_dataset.original_dataset, "原始训练集标签分布")
        
        # 分析平衡后的训练集标签分布
        analyze_label_distribution(train_dataset, "平衡后训练集标签分布")
    else:
        analyze_label_distribution(train_dataset, "训练集标签分布")
    
    # 分析验证集标签分布
    analyze_label_distribution(val_dataset, "验证集标签分布")
    
    # 分析测试集标签分布
    # 注意：测试集是按患者返回的，需要特殊处理
    test_labels = []
    for _, label, _ in test_dataset:
        test_labels.append(label.numpy())
    test_labels = np.array(test_labels)
    
    print("\n测试集标签分布:")
    for i in range(test_labels.shape[1]):
        positive_count = np.sum(test_labels[:, i])
        print(f"标签 {i+1}: 正例 {positive_count}, 负例 {len(test_labels) - positive_count}, 正例比例 {positive_count/len(test_labels):.2%}")
    
    # 查看训练集样本
    print("\n训练集样本:")
    show_images_and_labels(train_loader, num_images=5)
    
    # 查看测试集样本
    print("\n测试集样本:")
    show_images_and_labels(test_loader, num_images=3)