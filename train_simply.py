from build_dataset import create_dataloaders
import torch
import pandas as pd
import os
import numpy as np
import sys
import torch.nn as nn
from model_3_10 import TOpNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import json
from datetime import datetime

# 配置编码和环境变量
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 训练参数
batch_size = 8
num_epochs = 40
learning_rate = 5e-4
weight_decay = 1e-4
num_classes = 8
data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'
save_dir = 'D:/Fc25_07/FcccccUestc/results'

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建数据加载器
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
    data_dir=data_dir,
    label_dir=label_dir,
    batch_size=batch_size
)

def calculate_metrics(outputs, labels, threshold=0.3):
    """
    计算多标签分类的指标，以百分数形式表示
    """
    # 将输出转换为二进制预测
    sigmoid_outputs = torch.sigmoid(outputs)
    predictions = (sigmoid_outputs > threshold).float().cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 添加调试信息
    positive_preds = np.sum(predictions)
    positive_labels = np.sum(labels)
    print(f"预测中的正例数量: {positive_preds}, 标签中的正例数量: {positive_labels}")
    
    # 计算每个类别的TP, FP, FN
    true_positives = np.sum(np.logical_and(predictions == 1, labels == 1), axis=0)
    false_positives = np.sum(np.logical_and(predictions == 1, labels == 0), axis=0)
    false_negatives = np.sum(np.logical_and(predictions == 0, labels == 1), axis=0)
    
    # 计算精确率: TP / (TP + FP)
    precision = np.zeros(labels.shape[1])
    for i in range(labels.shape[1]):
        if true_positives[i] + false_positives[i] > 0:
            precision[i] = true_positives[i] / (true_positives[i] + false_positives[i]) * 100
        else:
            precision[i] = 0.0
    
    # 计算召回率: TP / (TP + FN)
    recall = np.zeros(labels.shape[1])
    for i in range(labels.shape[1]):
        if true_positives[i] + false_negatives[i] > 0:
            recall[i] = true_positives[i] / (true_positives[i] + false_negatives[i]) * 100
        else:
            recall[i] = 0.0
    
    # 计算准确率: (TP + TN) / (TP + TN + FP + FN)
    accuracy = np.mean(predictions == labels, axis=0) * 100
    
    # 计算平均指标
    precision_samples = np.mean(precision)
    recall_samples = np.mean(recall)
    hamming_accuracy = np.mean(accuracy)
    
    # 打印每个类别的指标
    for i in range(labels.shape[1]):
        print(f"label {i+1}: precison={precision[i]:.2f}%, recall={recall[i]:.2f}%, accuracy={accuracy[i]:.2f}%")
    
    print(f"precision: {precision_samples:.2f}%, recall: {recall_samples:.2f}%, accuracy: {hamming_accuracy:.2f}%")
    

    return precision_samples, recall_samples, hamming_accuracy

def train_epoch(model, dataloader, criterion, optimizer, device, threshold=0.5):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        # 梯度清零
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
    precision, recall, hamming_acc = calculate_metrics(all_outputs, all_labels, threshold)
    
    return epoch_loss, precision, recall, hamming_acc

def validate(model, dataloader, criterion, device, threshold=0.5):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="validating")
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
    precision, recall, hamming_acc = calculate_metrics(all_outputs, all_labels, threshold)
    
    return epoch_loss, precision, recall, hamming_acc

def test_model(model, test_loader, device, threshold=0.5, save_results=True):
    """测试模型并保存结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_left_preds = []
    all_right_preds = []
    all_combined_preds = []
    all_filenames = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="testing")
        for (left_img, right_img), labels, (left_name, right_name) in pbar:
            # 将图像移至设备
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            labels = labels.to(device).float()
            
            # 分别预测左右眼
            left_output = model(left_img)
            right_output = model(right_img)
            
            # 应用sigmoid获取概率
            left_probs = torch.sigmoid(left_output)
            right_probs = torch.sigmoid(right_output)
            
            # 合并左右眼预测结果 (取最大值)
            combined_probs = torch.max(left_probs, right_probs)
            
            # 应用阈值获取预测
            left_preds = (left_probs > threshold).float()
            right_preds = (right_probs > threshold).float()
            combined_preds = (combined_probs > threshold).float()
            
            # 收集结果
            all_left_preds.append(left_preds.cpu().numpy())
            all_right_preds.append(right_preds.cpu().numpy())
            all_combined_preds.append(combined_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_filenames.append((left_name[0], right_name[0]))
    
    # 转换为numpy数组
    all_left_preds = np.vstack(all_left_preds)
    all_right_preds = np.vstack(all_right_preds)
    all_combined_preds = np.vstack(all_combined_preds)
    all_labels = np.vstack(all_labels)
    
    # 计算指标
    left_precision = precision_score(all_labels, all_left_preds, average='samples', zero_division=0)
    left_recall = recall_score(all_labels, all_left_preds, average='samples', zero_division=0)
    left_f1 = 2 * (left_precision * left_recall) / (left_precision + left_recall + 1e-8)
    # left_exact_match = accuracy_score(all_labels, all_left_preds)
    
    right_precision = precision_score(all_labels, all_right_preds, average='samples', zero_division=0)
    right_recall = recall_score(all_labels, all_right_preds, average='samples', zero_division=0)
    right_f1 = 2 * (right_precision * right_recall) / (right_precision + right_recall + 1e-8)
    # right_exact_match = accuracy_score(all_labels, all_right_preds)
    
    combined_precision = precision_score(all_labels, all_combined_preds, average='samples', zero_division=0)
    combined_recall = recall_score(all_labels, all_combined_preds, average='samples', zero_division=0)
    combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall + 1e-8)
    # combined_exact_match = accuracy_score(all_labels, all_combined_preds)
    
    # 打印结果
    print("\n测试结果:")
    print(f"左眼 - 精确率: {left_precision:.4f}, 召回率: {left_recall:.4f}, F1: {left_f1:.4f}")
    print(f"右眼 - 精确率: {right_precision:.4f}, 召回率: {right_recall:.4f}, F1: {right_f1:.4f}")
    print(f"合并 - 精确率: {combined_precision:.4f}, 召回率: {combined_recall:.4f}, F1: {combined_f1:.4f}")
    
    # 保存结果
    if save_results:
        results = []
        class_names = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他疾病/异常"]  # 替换为实际类别名称
        
        for i in range(len(all_filenames)):
            left_name, right_name = all_filenames[i]
            patient_id = left_name.split('_')[0]
            
            # 获取左右眼和合并的预测结果
            left_pred = all_left_preds[i]
            right_pred = all_right_preds[i]
            combined_pred = all_combined_preds[i]
            true_label = all_labels[i]
            
            # 获取疾病名称
            left_diseases = [class_names[j] for j in range(len(left_pred)) if left_pred[j] == 1]
            right_diseases = [class_names[j] for j in range(len(right_pred)) if right_pred[j] == 1]
            combined_diseases = [class_names[j] for j in range(len(combined_pred)) if combined_pred[j] == 1]
            true_diseases = [class_names[j] for j in range(len(true_label)) if true_label[j] == 1]
            
            # 添加到结果列表
            results.append({
                "patient_id": patient_id,
                "left_eye": left_name,
                "right_eye": right_name,
                "left_prediction": left_diseases,
                "right_prediction": right_diseases,
                "combined_prediction": combined_diseases,
                "true_label": true_diseases,
                "correct": (combined_pred == true_label).all()
            })
        
        # 保存为JSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(save_dir, f"test_results_{timestamp}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"测试结果已保存到: {result_file}")
    
    return {
        "left_metrics": {
            "precision": left_precision,
            "recall": left_recall,
            "f1": left_f1,
          
        },
        "right_metrics": {
            "precision": right_precision,
            "recall": right_recall,
            "f1": right_f1,
          
        },
        "combined_metrics": {
            "precision": combined_precision,
            "recall": combined_recall,
            "f1": combined_f1,
            
        }
    }

def plot_metrics(train_metrics, val_metrics, save_path):
    """绘制训练和验证指标"""
    metrics = ['loss', 'precision', 'recall',  'hamming_acc', 'f1']
    titles = ['loss', 'precision', 'recall', 'hamming_acc', 'f1_score']
    
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
    train_metrics = {'loss': [], 'precision': [], 'recall': [], 'hamming_acc': [], 'f1': []}
    val_metrics = {'loss': [], 'precision': [], 'recall': [], 'hamming_acc': [], 'f1': []}
    
    # 早停设置
    patience = 10
    no_improve_epochs = 0
    
    # 动态阈值
    thresholds = [0.3, 0.4, 0.5]
    best_threshold = 0.3
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        train_loss, train_precision, train_recall,  train_hamming_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, threshold=best_threshold
        )
        
        # 计算F1分数
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        
        # 记录训练指标
        train_metrics['loss'].append(train_loss)
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['hamming_acc'].append(train_hamming_acc)
        train_metrics['f1'].append(train_f1)
        
        # 打印训练结果
        print(f"train - loss: {train_loss:.4f}%, precison: {train_precision:.4f}%, recall: {train_recall:.4f}%, "
              f"hamming_acc: {train_hamming_acc:.4f}%, F1: {train_f1:.4f}%")
        
        # 验证阶段 - 尝试不同阈值
        best_val_f1 = 0
        for threshold in thresholds:
            val_loss, val_precision, val_recall,val_hamming_acc = validate(
                model, val_loader, criterion, device, threshold=threshold
            )
            
            # 计算F1分数
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
            
            print(f"阈值 {threshold} - 验证 F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold = threshold
                best_val_metrics = (val_loss, val_precision, val_recall, val_hamming_acc)
        
        # 使用最佳阈值的结果
        val_loss, val_precision, val_recall, val_hamming_acc = best_val_metrics
        val_f1 = best_val_f1
        
        print(f"最佳阈值: {best_threshold}")
        
        # 记录验证指标
        val_metrics['loss'].append(val_loss)
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['hamming_acc'].append(val_hamming_acc)
        val_metrics['f1'].append(val_f1)
        
        # 打印验证结果
        print(f"valid - loss: {val_loss:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}, "
              f"hamming_acc: {val_hamming_acc:.4f}, F1: {val_f1:.4f}")
        
        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_f1)
        else:
            scheduler.step()
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"保存新的最佳模型，F1: {val_f1:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"验证性能未提升，已经 {no_improve_epochs}/{patience} 轮")
            
        # 早停
        if no_improve_epochs >= patience:
            print(f"早停：验证性能已经 {patience} 轮未提升")
            break
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 绘制训练和验证指标
    plot_metrics(train_metrics, val_metrics, os.path.join(save_dir, 'training_metrics.png'))
    
    # 保存训练和验证指标到CSV
    pd.DataFrame(train_metrics).to_csv(os.path.join(save_dir, 'train_metrics.csv'), index=False)
    pd.DataFrame(val_metrics).to_csv(os.path.join(save_dir, 'val_metrics.csv'), index=False)
    
    return model, train_metrics, val_metrics, best_threshold


if __name__ == "__main__":
    # 初始化模型
    model = TOpNet()
    
    # 计算类别权重以处理不平衡问题
    all_labels = []
    for _, label in train_dataset:
        all_labels.append(label.numpy())
    all_labels = np.vstack(all_labels)
    pos_counts = np.sum(all_labels, axis=0)
    neg_counts = len(all_labels) - pos_counts
    pos_weights = torch.FloatTensor(neg_counts / (pos_counts + 1e-8)).to(device)
    
    print("类别分布:")
    for i, (pos, neg) in enumerate(zip(pos_counts, neg_counts)):
        print(f"类别 {i}: 正例 {pos}, 负例 {neg}, 权重 {neg/(pos+1e-8):.2f}")
    
    # 定义损失函数和优化器 - 使用带权重的损失函数
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # 使用更适合的优化器设置
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器 - 使用更温和的衰减
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5,  # 更温和的衰减
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # 训练和验证模型
    model, train_metrics, val_metrics, best_threshold = train_val(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # 保存模型
    model_save_path = os.path.join(save_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 保存最佳阈值
    with open(os.path.join(save_dir, 'best_threshold.txt'), 'w') as f:
        f.write(str(best_threshold))
    
    # 测试模型
    print("\n开始测试模型...")
    test_results = test_model(model, test_loader, device, threshold=best_threshold)
    
    # 保存测试结果
    test_metrics = {
        "left_eye": test_results["left_metrics"],
        "right_eye": test_results["right_metrics"],
        "combined": test_results["combined_metrics"],
        "threshold": best_threshold
    }
    
    # 将测试指标保存为JSON
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=4)
    
    print("训练、验证和测试完成！")