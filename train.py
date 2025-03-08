from build_dataset import CustomDataset, TransformSubset
import torch
from torchvision import transforms, datasets
import pandas as pd
import os
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import numpy as np 
import sys
import torch.nn as nn
from model import TanNet
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    multilabel_confusion_matrix
)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'
# Excel 文件路径
generator = torch.Generator().manual_seed(42)

# 将 DataFrame 保存为 CSV 文件
df = pd.read_csv(label_dir, header=0, encoding='utf-8')

batch_size = 32

train_transforms = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    # transforms.ColorJitter(brightness=1.1, contrast=1.5, saturation=0.8),       # 可根据训练结果调节
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=35),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset = CustomDataset(data_dir=data_dir, label_dir=label_dir, transform=None)
data_len = len(dataset)


train_size = int(0.8 * data_len)
val_size = data_len - train_size
train_indices, val_indices = torch.utils.data.random_split(
    dataset=dataset, lengths=[train_size, val_size], generator=generator
)

train_dataset = TransformSubset(train_indices, transform=train_transforms)
val_dataset = TransformSubset(val_indices, transform=valid_transforms)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

criterion = nn.CrossEntropyLoss()
model = TanNet(8)

# 损失函数和采样器
def get_loss_function(pos_weights=None):
    if pos_weights is not None:
        pos_weights = torch.tensor(pos_weights).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)

def get_samplers(dataset):
    # 获取所有标签并堆叠为二维张量 (num_samples, num_classes)
    labels = [sample[1] for sample in dataset]
    all_labels = torch.stack(labels)  # 假设每个标签是形状(num_classes,)的张量
    
    # 统计每个类别的正样本数
    class_counts = all_labels.sum(dim=0).cpu().numpy()  # 形状(num_classes,)
    
    # 添加平滑处理防止除零
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    # 计算类别权重（注意：这里需要反转权重逻辑）
    class_weights = {cls_idx: 1.0 / count for cls_idx, count in enumerate(class_counts)}
    
    # 生成样本权重（每个样本的权重是其所有正类权重的平均）
    sample_weights = []
    for label in labels:
        # 转换为numpy数组处理
        label_np = label.cpu().numpy()
        positive_classes = np.where(label_np == 1)[0]
        
        if len(positive_classes) == 0:
            # 处理没有正类的情况
            sample_weights.append(1.0)
        else:
            # 计算平均权重
            weight = np.mean([class_weights[cls] for cls in positive_classes])
            sample_weights.append(weight)
    
    # 创建采样器
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=2*len(dataset),
        replacement=True
    )

# 训练验证框架
class Trainer:
    def __init__(self, model, train_loader, val_loader, num_classes, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # 获取类别权重
        all_labels = torch.cat([y for _, y in train_loader.dataset])
        pos_counts = all_labels.sum(dim=0)
        pos_weights = (len(all_labels) - pos_counts) / pos_counts
        
        self.criterion = get_loss_function(pos_weights)
        self.optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=3)
        
        # 训练记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': []
        }

        # 初始化TensorBoard
        self.writer = SummaryWriter(
            f"runs/{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # 记录模型计算图
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        self.writer.add_graph(model, dummy_input)

    def _compute_metrics(self, y_true, y_pred, threshold=0.6):
        y_pred_bin = (y_pred > threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred_bin)
        precision = precision_score(y_true, y_pred_bin, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred_bin, average='samples', zero_division=0)
        
        return acc, precision, recall

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            targets = targets.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().detach().numpy())
            all_targets.append(targets.cpu().numpy())
        
        # 计算指标
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        acc, precision, recall = self._compute_metrics(all_targets, all_preds)
        
        # 记录训练指标
        self.writer.add_scalar('Loss/train', epoch_loss, 50)
        self.writer.add_scalar('Accuracy/train', acc, 50)
        self.writer.add_scalar('Precision/train', precision, 50)
        self.writer.add_scalar('Recall/train', recall, 50)
        
        return epoch_loss/len(self.train_loader), acc, precision, recall

    def _validate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                epoch_loss += loss.item()
                # 使用sigmoid转换概率
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 合并所有批次结果
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # 应用阈值0.6
        y_pred = (all_preds > 0.6).astype(int)
        y_true = all_targets

        # 计算全局指标
        val_loss = epoch_loss / len(self.val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        val_precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        val_recall = recall_score(y_true, y_pred, average='samples', zero_division=0)
        
        # 计算每个类别的指标
        class_metrics = []
        for class_idx in range(self.num_classes):
            class_precision = precision_score(
                y_true[:, class_idx], 
                y_pred[:, class_idx], 
                zero_division=0
            )
            class_recall = recall_score(
                y_true[:, class_idx], 
                y_pred[:, class_idx], 
                zero_division=0
            )
            class_metrics.append((class_precision, class_recall))

        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'class_metrics': class_metrics,
            'y_true': y_true,
            'y_pred': y_pred
        }

    def print_detailed_metrics(self, val_results):
        """打印详细验证指标"""
        print("\nValidation Metrics:")
        print(f"Loss: {val_results['loss']:.4f}")
        print(f"Accuracy: {val_results['accuracy']:.4f}")
        print(f"Precision (samples): {val_results['precision']:.4f}")
        print(f"Recall (samples): {val_results['recall']:.4f}")
        
        # 打印每个类别的指标
        print("\nPer-Class Metrics:")
        for idx, (prec, rec) in enumerate(val_results['class_metrics']):
            print(f"Class {idx}: Precision={prec:.4f}, Recall={rec:.4f}")

    def plot_confusion_matrix(self, val_results, class_names=None, max_classes=6):
        """可视化混淆矩阵（最多显示前6个类别）"""
        y_true = val_results['y_true']
        y_pred = val_results['y_pred']
        
        # 生成多标签混淆矩阵
        cm = multilabel_confusion_matrix(y_true, y_pred)
        
        # 设置类别名称
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        # 限制显示类别数量
        display_classes = min(max_classes, self.num_classes)
        
        plt.figure(figsize=(15, 10))
        for i in range(display_classes):
            plt.subplot(2, 3, i+1)
            sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.title(f'{class_names[i]}\nPrecision: {val_results["class_metrics"][i][0]:.2f}'
                      f'  Recall: {val_results["class_metrics"][i][1]:.2f}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

    def train(self, epochs):
        best_score = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练阶段
            train_loss, train_acc, train_prec, train_rec = self._train_epoch()
            
            # 验证阶段（返回完整结果）
            val_results = self._validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_results['recall'])
            
            # 记录历史数据
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_results['accuracy'])
            self.history['train_precision'].append(train_prec)
            self.history['val_precision'].append(val_results['precision'])
            self.history['train_recall'].append(train_rec)
            self.history['val_recall'].append(val_results['recall'])
            
            # 打印详细指标
            self.print_detailed_metrics(val_results)
            
            # 保存最佳模型
            if val_results['recall'] > best_score:
                best_score = val_results['recall']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Saved new best model!")
            
            # 每5个epoch可视化一次
            if (epoch+1) % 5 == 0:
                self.plot_confusion_matrix(val_results)
                self.plot_metrics()

    def _log_confusion_matrix(self, y_true, y_pred, epoch):
        """将混淆矩阵记录到TensorBoard"""
        cm = multilabel_confusion_matrix(y_true, y_pred)
        
        for class_idx in range(min(6, self.num_classes)):  # 最多显示6个类别
            fig = plt.figure(figsize=(6, 6))
            sns.heatmap(cm[class_idx], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Class {class_idx} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # 将matplotlib图像转为TensorBoard可识别的格式
            self.writer.add_figure(
                f'Confusion Matrix/Class_{class_idx}', 
                fig, 
                global_step=epoch
            )
            plt.close(fig)

    
    def _log_histograms(self, epoch):
        """记录权重和梯度直方图"""
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def _log_pr_curves(self, y_true, y_pred_probs, epoch):
        """记录PR曲线"""
        for class_idx in range(min(6, self.num_classes)):  # 最多显示6个类别
            true_labels = y_true[:, class_idx]
            pred_probs = y_pred_probs[:, class_idx]
            self.writer.add_pr_curve(
                f'PR Curve/Class_{class_idx}',
                true_labels,
                pred_probs,
                global_step=epoch
            )

if __name__ == "__main__":
    # 初始化模型
    num_classes = 8  
    model = TanNet(num_classes=num_classes)
    
    # 创建带采样的DataLoader
    train_sampler = get_samplers(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device
    )
    
    # 开始训练
    trainer.train(epochs=50)
    
    # 可视化结果
    trainer.plot_metrics()
    class_names = ["肺炎", "癌症", "正常", ...]  # 根据实际标签顺序
    trainer.plot_confusion_matrix(threshold=0.6, class_names=class_names)
    
    