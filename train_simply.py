from build_dataset import CustomDataset, TransformSubset
import torch
from torchvision import transforms
import pandas as pd
import os
import numpy as np
import sys
import torch.nn as nn
from model_lightest import TannetLigtht
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# 配置编码和环境变量
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用 {device} 设备")

# 数据路径
data_dir = '/root/autodl-tmp/Fc25_07/FcccccUestc/Training_data'
label_dir = '/root/autodl-tmp/Fc25_07/FcccccUestc/total_data.csv'
save_dir = '/root/autodl-tmp/Fc25_07/FcccccUestc/results'
os.makedirs(save_dir, exist_ok=True)

# 设置随机种子
generator = torch.Generator().manual_seed(42)

# 读取标签数据
df = pd.read_csv(label_dir, header=0, encoding='utf-8')

# 训练参数
batch_size = 4
num_epochs = 35
learning_rate = 5e-5
weight_decay = 1e-4
num_classes = 8

# 数据增强
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
dataset = CustomDataset(data_dir=data_dir, label_dir=label_dir, transform=None)
data_len = len(dataset)

# 划分训练集和验证集
train_size = int(0.8 * data_len)
val_size = data_len - train_size
train_indices, val_indices = torch.utils.data.random_split(
    dataset=dataset, lengths=[train_size, val_size], generator=generator
)

# 应用数据转换
train_dataset = TransformSubset(train_indices, transform=train_transforms)
val_dataset = TransformSubset(val_indices, transform=valid_transforms)

# 创建数据加载器
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
valid_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def calculate_metrics(outputs, labels, threshold=0.5):
    """
    计算多标签分类的指标
    
    Args:
        outputs: 模型输出的预测值
        labels: 真实标签
        threshold: 二分类阈值
    
    Returns:
        precision, recall, accuracy: 精确率、召回率、准确率
    """
    # 将输出转换为二进制预测
    predictions = (torch.sigmoid(outputs) > threshold).float().cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 计算样本级别的精确率、召回率和准确率
    precision = precision_score(labels, predictions, average='samples', zero_division=0)
    recall = recall_score(labels, predictions, average='samples', zero_division=0)
    
    # 计算准确率 (完全匹配的样本比例)
    exact_match_accuracy = accuracy_score(labels, predictions)
    
    # 计算汉明准确率 (正确预测的标签比例)
    hamming_accuracy = (predictions == labels).mean()
    
    return precision, recall, exact_match_accuracy, hamming_accuracy

def train_epoch(model, dataloader, criterion, optimizer, device, threshold=0.5):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="训练中")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        all_outputs.append(outputs.detach())
        all_labels.append(labels)
        
        # 更新进度条
        pbar.set_postfix({"batch_loss": loss.item()})
    
    # 计算平均损失
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # 合并所有批次的输出和标签
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    precision, recall, exact_match_acc, hamming_acc = calculate_metrics(all_outputs, all_labels, threshold)
    
    return epoch_loss, precision, recall, exact_match_acc, hamming_acc

def validate(model, dataloader, criterion, device, threshold=0.5):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="验证中")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)
            
            # 更新进度条
            pbar.set_postfix({"batch_loss": loss.item()})
    
    # 计算平均损失
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # 合并所有批次的输出和标签
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    precision, recall, exact_match_acc, hamming_acc = calculate_metrics(all_outputs, all_labels, threshold)
    
    return epoch_loss, precision, recall, exact_match_acc, hamming_acc

def plot_metrics(train_metrics, val_metrics, save_path):
    """绘制训练和验证指标"""
    metrics = ['loss', 'precision', 'recall', 'exact_match_acc', 'hamming_acc']
    titles = ['损失', '精确率', '召回率', '完全匹配准确率', '汉明准确率']
    
    plt.figure(figsize=(15, 12))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(3, 2, i+1)
        plt.plot(train_metrics[metric], label=f'训练 {title}')
        plt.plot(val_metrics[metric], label=f'验证 {title}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'{title} 曲线')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_val(model, criterion, optimizer, scheduler, num_epochs=40):
    """训练和验证模型"""
    model.to(device)
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 记录训练和验证指标
    train_metrics = {'loss': [], 'precision': [], 'recall': [], 'exact_match_acc': [], 'hamming_acc': [], 'f1': []}
    val_metrics = {'loss': [], 'precision': [], 'recall': [], 'exact_match_acc': [], 'hamming_acc': [], 'f1': []}
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        train_loss, train_precision, train_recall, train_exact_acc, train_hamming_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, device, threshold=0.25
        )
        
        # 计算F1分数
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        
        # 记录训练指标
        train_metrics['loss'].append(train_loss)
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['exact_match_acc'].append(train_exact_acc)
        train_metrics['hamming_acc'].append(train_hamming_acc)
        train_metrics['f1'].append(train_f1)
        
        # 打印训练结果
        print(f"训练 - 损失: {train_loss:.4f}, 精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, "
              f"完全匹配准确率: {train_exact_acc:.4f}, 汉明准确率: {train_hamming_acc:.4f}, F1: {train_f1:.4f}")
        
        # 验证阶段
        val_loss, val_precision, val_recall, val_exact_acc, val_hamming_acc = validate(
            model, valid_dataloader, criterion, device, threshold=0.25
        )
        
        # 计算F1分数
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        # 记录验证指标
        val_metrics['loss'].append(val_loss)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['exact_match_acc'].append(val_exact_acc)
        val_metrics['hamming_acc'].append(val_hamming_acc)
        val_metrics['f1'].append(val_f1)
        
        # 打印验证结果
        print(f"验证 - 损失: {val_loss:.4f}, 精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}, "
              f"完全匹配准确率: {val_exact_acc:.4f}, 汉明准确率: {val_hamming_acc:.4f}, F1: {val_f1:.4f}")
        
        # 更新学习率
        scheduler.step(val_f1)
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"保存新的最佳模型，F1: {val_f1:.4f}")
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 绘制训练和验证指标
    plot_metrics(train_metrics, val_metrics, os.path.join(save_dir, 'training_metrics.png'))
    
    # 保存训练和验证指标到CSV
    pd.DataFrame(train_metrics).to_csv(os.path.join(save_dir, 'train_metrics.csv'), index=False)
    pd.DataFrame(val_metrics).to_csv(os.path.join(save_dir, 'val_metrics.csv'), index=False)
    
    return model, train_metrics, val_metrics

if __name__ == "__main__":
    # 初始化模型
    model = TannetLigtht(num_classes=num_classes)
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    # 训练和验证模型
    model, train_metrics, val_metrics = train_val(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # 保存模型
    model_save_path = os.path.join(save_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")