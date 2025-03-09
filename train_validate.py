import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score, average_precision_score
from model_lightest import TannetLigtht
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from build_dataset import CustomDataset  

def train_epoch(model, dataloader, criterion, optimizer, device, threshold=0.5):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="训练中")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device).float()  # 确保标签是浮点型，适用于多标签
        
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
        preds = (torch.sigmoid(outputs) > threshold).float().cpu().numpy()  # 使用sigmoid激活并应用阈值
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({"batch_loss": loss.item()})
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # 计算多标签分类指标
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    
    # 计算每个样本的准确率（样本级别的准确率）
    sample_accuracy = (all_preds == all_labels).mean(axis=1).mean()
    
    return epoch_loss, sample_accuracy, precision, recall, f1, all_preds, all_labels

def validate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_outputs = []  # 存储原始输出，用于计算AP
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="验证中")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).float()  # 确保标签是浮点型
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > threshold).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({"batch_loss": loss.item()})
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    
    # 计算多标签分类指标
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    
    # 计算每个样本的准确率
    sample_accuracy = (all_preds == all_labels).mean(axis=1).mean()
    
    # 计算平均精度AP
    ap = average_precision_score(all_labels, all_outputs, average='samples')
    
    return epoch_loss, sample_accuracy, precision, recall, f1, ap, all_preds, all_labels

def plot_multilabel_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """为每个类别绘制混淆矩阵"""
    num_classes = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 假设有8个类别
    axes = axes.flatten()
    
    for i in range(num_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
        axes[i].set_title(f'类别: {class_names[i]}')
        axes[i].set_xticklabels(['负例', '正例'])
        axes[i].set_yticklabels(['负例', '正例'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics(train_metrics, val_metrics, save_dir=None):
    """绘制训练和验证指标"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    plt.figure(figsize=(15, 12))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        plt.plot(train_metrics[metric], label=f'训练 {metric}')
        plt.plot(val_metrics[metric], label=f'验证 {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} 曲线')
        plt.legend()
    
    # 额外绘制AP曲线（仅验证集有）
    if 'ap' in val_metrics:
        plt.subplot(3, 2, 6)
        plt.plot(val_metrics['ap'], label='验证 AP')
        plt.xlabel('Epoch')
        plt.ylabel('AP')
        plt.title('平均精度曲线')
        plt.legend()
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()

def main():
    # 设置参数
    num_classes = 8  # 根据您的实际类别数修改
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = CustomDataset(data_path="d:/Fc25_07/FcccccUestc/data/train", transform=transform)
    val_dataset = CustomDataset(data_path="d:/Fc25_07/FcccccUestc/data/val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 类别名称
    class_names = [f"类别{i}" for i in range(num_classes)]  # 替换为实际的类别名称
    
    # 初始化模型
    model = TannetLigtht(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器 - 多标签分类使用BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 创建保存结果的目录
    save_dir = "d:/Fc25_07/FcccccUestc/results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录训练和验证指标
    train_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'ap': []}
    
    best_val_f1 = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        train_loss, train_acc, train_prec, train_recall, train_f1, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证阶段
        val_loss, val_acc, val_prec, val_recall, val_f1, val_ap, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录指标
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_acc)
        train_metrics['precision'].append(train_prec)
        train_metrics['recall'].append(train_recall)
        train_metrics['f1'].append(train_f1)
        
        val_metrics['loss'].append(val_loss)
        val_metrics['accuracy'].append(val_acc)
        val_metrics['precision'].append(val_prec)
        val_metrics['recall'].append(val_recall)
        val_metrics['f1'].append(val_f1)
        val_metrics['ap'].append(val_ap)
        
        # 打印当前epoch的结果
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, 精确率: {train_prec:.4f}, 召回率: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, 精确率: {val_prec:.4f}, 召回率: {val_recall:.4f}, F1: {val_f1:.4f}, AP: {val_ap:.4f}")
        
        # 每个epoch绘制并保存混淆矩阵
        plot_multilabel_confusion_matrix(
            train_labels, train_preds, class_names, 
            save_path=os.path.join(save_dir, f"train_confusion_matrix_epoch_{epoch+1}.png")
        )
        
        plot_multilabel_confusion_matrix(
            val_labels, val_preds, class_names, 
            save_path=os.path.join(save_dir, f"val_confusion_matrix_epoch_{epoch+1}.png")
        )
        
        # 保存最佳模型 (使用F1分数作为指标)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"模型已保存! 验证F1分数: {val_f1:.4f}")
    
    # 绘制并保存训练过程中的指标变化
    plot_metrics(train_metrics, val_metrics, save_dir=save_dir)
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    
    # 保存训练和验证指标到CSV
    train_df = pd.DataFrame(train_metrics)
    val_df = pd.DataFrame(val_metrics)
    
    train_df.to_csv(os.path.join(save_dir, "train_metrics.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val_metrics.csv"), index=False)
    
    print("训练完成!")

if __name__ == "__main__":
    main()