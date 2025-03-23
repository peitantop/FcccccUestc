import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_new01 import TOpNet
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 配置参数
model_weights_path = 'D:/Fc25_07/FcccccUestc/results/model_weights.pth'
threshold = 0.3
num_classes = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 类别标签映射
class_names = [
    "正常", "糖尿病", "青光眼", "白内障", 
    "AMD", "高血压", "近视", "其他疾病/异常"
]

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    return image_tensor

# 加载模型
def load_model(weights_path):
    model = TOpNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 预测函数
def predict(model, left_eye_path, right_eye_path, threshold=0.3):
    # 预处理图像
    left_img = preprocess_image(left_eye_path).to(device)
    right_img = preprocess_image(right_eye_path).to(device)
    
    # 进行预测
    with torch.no_grad():
        left_output = model(left_img)
        right_output = model(right_img)
        
        # 应用sigmoid获取概率
        left_probs = torch.sigmoid(left_output)
        right_probs = torch.sigmoid(right_output)
        
        # 合并左右眼预测结果（取最大值）
        combined_probs = torch.max(left_probs, right_probs)
        
        # 应用阈值获取预测标签
        predictions = (combined_probs > threshold).float().cpu().numpy()[0]
        probabilities = combined_probs.cpu().numpy()[0]
    
    # 返回预测结果和概率
    results = []
    for i in range(num_classes):
        results.append({
            "class": class_names[i],
            "probability": float(probabilities[i]),
            "predicted": bool(predictions[i])
        })
    
    return results

if __name__ == "__main__":
    # 加载模型
    model = load_model(model_weights_path)
    print(f"模型已加载: {model_weights_path}")
    
    # 示例：预测单对图像
    left_eye_path = input("请输入左眼图像路径: ")
    right_eye_path = input("请输入右眼图像路径: ")
    
    if os.path.exists(left_eye_path) and os.path.exists(right_eye_path):
        results = predict(model, left_eye_path, right_eye_path, threshold)
        
        print("\n预测结果:")
        print("-" * 50)
        for result in results:
            status = "检测到" if result["predicted"] else "未检测到"
            print(f"{result['class']}: {result['probability']:.4f} ({status})")
        
        # 输出检测到的所有类别
        detected_classes = [result["class"] for result in results if result["predicted"]]
        if detected_classes:
            print("\n检测到的类别:", ", ".join(detected_classes))
        else:
            print("\n未检测到任何类别")
    else:
        print("错误：图像文件不存在，请检查路径")

# 批量预测函数（如果需要）
def batch_predict(model, image_pairs_folder, output_file=None):
    """
    对文件夹中的图像对进行批量预测
    
    参数:
    - model: 加载的模型
    - image_pairs_folder: 包含左右眼图像对的文件夹
    - output_file: 输出结果的文件路径（可选）
    
    文件夹结构假设为:
    image_pairs_folder/
        patient1_left.jpg
        patient1_right.jpg
        patient2_left.jpg
        patient2_right.jpg
        ...
    """
    results = {}
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_pairs_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 按患者ID分组
    patient_ids = set()
    for img_file in image_files:
        patient_id = img_file.split('_')[0]  # 假设文件名格式为 "patientID_eye.jpg"
        patient_ids.add(patient_id)
    
    for patient_id in patient_ids:
        left_file = next((f for f in image_files if f.startswith(f"{patient_id}_left")), None)
        right_file = next((f for f in image_files if f.startswith(f"{patient_id}_right")), None)
        
        if left_file and right_file:
            left_path = os.path.join(image_pairs_folder, left_file)
            right_path = os.path.join(image_pairs_folder, right_file)
            
            prediction = predict(model, left_path, right_path, threshold)
            results[patient_id] = prediction
            
            print(f"已完成 {patient_id} 的预测")
    
    # 保存结果到文件（如果指定）
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"预测结果已保存到 {output_file}")
    
    return results

# 使用批量预测的示例（取消注释以使用）

# if __name__ == "__main__":
    # 加载模型
    # model = load_model(model_weights_path)
    #
    # # 批量预测
    # image_folder = "C:/Users/peit8/Desktop/服创/kaggle_datas/On-site Test Set/Images"
    # output_path = "D:/Fc25_07/FcccccUestc/results/batch_predictions.json"
    #
    # batch_results = batch_predict(model, image_folder, output_path)
