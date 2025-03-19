from build_dataset import create_dataloaders
import torch
import pandas as pd
import os
import numpy as np
import sys
import torch.nn as nn
from model_new01 import TOpNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

batch_size = 4
num_epochs = 150
learning_rate = 1e-5
weight_decay = 1e-4
num_classes = 8
data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'
save_dir = 'D:/Fc25_07/FcccccUestc/results'
threshold = 0.3

os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
    data_dir=data_dir, label_dir=label_dir, batch_size=batch_size
)

def calculate_metrics(outputs, labels, threshold=0.3):
    sigmoid_outputs = torch.sigmoid(outputs)
    predictions = (sigmoid_outputs > threshold).float().cpu().numpy()
    labels = labels.cpu().numpy()
    
    precision = average_precision_score(labels, predictions, average='samples') * 100
    recall = recall_score(labels, predictions, average='samples', zero_division=0) * 100

    label_accuracies = []
    for i in range(labels.shape[1]):
        label_acc = accuracy_score(labels[:, i], predictions[:, i])
        label_accuracies.append(label_acc)
    accuracy = np.mean(label_accuracies) * 100
    
    return precision, recall, accuracy

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        all_outputs.append(outputs.detach())
        all_labels.append(labels)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    precision, recall, accuracy = calculate_metrics(all_outputs, all_labels)
    print(f"Train - Loss: {epoch_loss:.4f}, Precision: {precision:.2f}%, Recall: {recall:.2f}%, Accuracy: {accuracy:.2f}%")
    
    return epoch_loss, precision, recall, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs)
            all_labels.append(labels)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    precision, recall, accuracy = calculate_metrics(all_outputs, all_labels)
    print(f"Valid - Loss: {epoch_loss:.4f}, Precision: {precision:.2f}%, Recall: {recall:.2f}%, Accuracy: {accuracy:.2f}%")
    
    return epoch_loss, precision, recall, accuracy



def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_combined_preds = []
    
    with torch.no_grad():
        for (left_img, right_img), labels, _ in tqdm(test_loader, desc="testing"):
            left_img, right_img = left_img.to(device), right_img.to(device)
            labels = labels.to(device).float()
            
            left_output = model(left_img)
            right_output = model(right_img)
            
            left_probs = torch.sigmoid(left_output)
            right_probs = torch.sigmoid(right_output)
            combined_probs = torch.max(left_probs, right_probs)
            combined_preds = (combined_probs > threshold).float()
            
            all_combined_preds.append(combined_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_combined_preds = np.vstack(all_combined_preds)
    all_labels = np.vstack(all_labels)
    
    precision = precision_score(all_labels, all_combined_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_combined_preds, average='samples', zero_division=0)
    
    # 修改：计算标签级准确率
    label_accuracies = []
    for i in range(all_labels.shape[1]):
        label_acc = accuracy_score(all_labels[:, i], all_combined_preds[:, i])
        label_accuracies.append(label_acc)
    accuracy = np.mean(label_accuracies)
    
    print(f"\n测试结果:\n精确率: {precision:.4f}, 召回率: {recall:.4f}, 准确率: {accuracy:.4f}")
    return {"precision": precision, "recall": recall, "accuracy": accuracy}

def plot_metrics(train_metrics, val_metrics, save_path):
    metrics = ['loss', 'precision', 'recall', 'accuracy']
    titles = ['Loss', 'Precision', 'Recall', 'Accuracy']
    
    plt.figure(figsize=(15, 12))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i+1)
        plt.plot(train_metrics[metric], label='Training')
        plt.plot(val_metrics[metric], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'{title} Curve')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_val(model, criterion, optimizer, scheduler, num_epochs=40):
    model.to(device)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(save_dir, 'runs', current_time)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志保存在: {log_dir}")

    writer = SummaryWriter(os.path.join(save_dir, 'runs'))
    best_score = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_metrics = {'loss': [], 'precision': [], 'recall': [], 'accuracy': []}
    val_metrics = {'loss': [], 'precision': [], 'recall': [], 'accuracy': []}

    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_precision, train_recall, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_precision, val_recall, val_accuracy = validate(
            model, val_loader, criterion, device
        )
        
        # 记录指标到 tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # 更新指标历史记录
        train_metrics['loss'].append(train_loss)
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['accuracy'].append(train_accuracy)
        
        val_metrics['loss'].append(val_loss)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['accuracy'].append(val_accuracy)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_accuracy)
        else:
            scheduler.step()
        
        # 计算综合得分
        current_score = (val_precision + val_recall + val_accuracy) / 3
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        
        # 更新最佳模型
        if current_score > best_score:
            best_score = current_score
            best_model_wts = copy.deepcopy(model.state_dict())
            # 保存最佳验证指标
            best_val_metrics = {
                'epoch': epoch + 1,
                'loss': val_loss,
                'precision': val_precision,
                'recall': val_recall,
                'accuracy': val_accuracy,
                'combined_score': current_score
            }
            with open(os.path.join(save_dir, 'valid_metrics.json'), 'w') as f:
                json.dump(best_val_metrics, f, indent=4)
    
    writer.close()
    model.load_state_dict(best_model_wts)
    plot_metrics(train_metrics, val_metrics, os.path.join(save_dir, 'training_metrics.png'))
    return model, train_metrics, val_metrics
if __name__ == "__main__":
    model = TOpNet()
    model.load_state_dict(torch.load("D:/Fc25_07/FcccccUestc/results/model_weights.pth", weights_only=True))
    # # pos_counts = train_dataset.labels.sum(axis=0)
    # # neg_counts = len(train_dataset) - pos_counts
    # # pos_weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float).to(device)
    # pos_counts = train_dataset.labels.sum(axis=0)  # [C]
    # neg_counts = train_dataset.labels.shape[0] - pos_counts  # [C]
    # total_counts = pos_counts + neg_counts
    # beta = 0.999  # 控制平滑强度
    # effective_num = 1.0 - np.power(beta, pos_counts)
    # pos_weights = (1.0 - beta) / (effective_num + 1e-6)
    # pos_weights = torch.FloatTensor(pos_weights).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    
    model, train_metrics, val_metrics = train_val(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
    
    test_results = test_model(model, test_loader, device)
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_results, f, indent=4)