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
from datetime import datetime

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

batch_size = 8
num_epochs = 20
learning_rate = 1e-4
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
    best_acc = 0.0
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
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    plot_metrics(train_metrics, val_metrics, os.path.join(save_dir, 'training_metrics.png'))
    return model, train_metrics, val_metrics

if __name__ == "__main__":
    model = TOpNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    
    model, train_metrics, val_metrics = train_val(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
    
    test_results = test_model(model, test_loader, device)
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_results, f, indent=4)