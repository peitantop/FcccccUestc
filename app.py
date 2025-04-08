from flask import Flask, request, jsonify
from predict import load_model, predict
from flask_cors import CORS
import os
import zipfile
import tempfile
import uuid

app = Flask(__name__)
CORS(app)  

model_weights_path = 'C:/Users/peit8/Desktop/fuchuang/model_results/bestmodel2/model_weights.pth'
model = load_model(model_weights_path)


@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        all_results = {}
        temp_dir = 'd:/Fc25_07/FcccccUestc/temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        # 检查是否直接上传了两张图片（而不是zip文件）
        image_files = request.files.getlist('images')
        if len(image_files) == 2:
            # 生成唯一的病人ID
            patient_id = f"patient_{uuid.uuid4().hex[:8]}"
            
            # 创建病人专属的临时目录
            patient_temp_dir = os.path.join(temp_dir, patient_id)
            os.makedirs(patient_temp_dir, exist_ok=True)
            
            try:
                # 保存两张图片
                image_paths = []
                for i, img_file in enumerate(image_files):
                    file_ext = os.path.splitext(img_file.filename)[1]
                    image_path = os.path.join(patient_temp_dir, f"eye_{i}{file_ext}")
                    img_file.save(image_path)
                    image_paths.append(image_path)
                
                # 进行预测
                results = predict(model, image_paths[0], image_paths[1])
                
                # 处理预测结果
                if isinstance(results, dict) and 'prediction_results' in results:
                    all_results[patient_id] = {
                        "diseases": results["prediction_results"]["diseases"]
                    }
                else:
                    formatted_results = {"diseases": {}}
                    for result in results:
                        disease_name = result["class"]
                        prob_value = float(result["probability"])
                        predicted = bool(result["predicted"])
                        
                        formatted_results["diseases"][disease_name] = {
                            "probability": prob_value,
                            "status": 1 if predicted else 0
                        }
                    all_results[patient_id] = formatted_results
            
            except Exception as e:
                all_results[patient_id] = {"error": str(e)}
                
            finally:
                # 清理临时文件
                if os.path.exists(patient_temp_dir):
                    for file in os.listdir(patient_temp_dir):
                        try:
                            os.remove(os.path.join(patient_temp_dir, file))
                        except:
                            pass
                    try:
                        os.rmdir(patient_temp_dir)
                    except:
                        pass
                        
            return jsonify({"prediction_results": all_results})
        
        # 获取所有上传的zip文件
        zip_files = request.files.getlist('file')
        
        if not zip_files:
            return jsonify({"error": "未提供zip文件或图片文件"}), 400
        
        # 处理每个上传的zip文件
        for zip_file in zip_files:
            try:
                patient_id = os.path.splitext(zip_file.filename)[0]  # 使用文件名作为病人ID
                
                # 创建病人专属的临时目录
                patient_temp_dir = os.path.join(temp_dir, patient_id)
                os.makedirs(patient_temp_dir, exist_ok=True)
                
                # 保存并解压zip文件
                zip_path = os.path.join(patient_temp_dir, f"{patient_id}.zip")
                zip_file.save(zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(patient_temp_dir)
                
                # 获取解压后的文件和文件夹
                all_items = os.listdir(patient_temp_dir)
                image_files = []
                
                # 检查是否存在子文件夹
                for item in all_items:
                    item_path = os.path.join(patient_temp_dir, item)
                    if os.path.isdir(item_path) and item != '__MACOSX':  # 排除 Mac 系统生成的隐藏文件夹
                        # 如果是文件夹，从文件夹中获取图片
                        subfolder_items = os.listdir(item_path)
                        for subitem in subfolder_items:
                            if subitem.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_files.append(os.path.join(item_path, subitem))
                    elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 如果是图片文件，直接添加
                        image_files.append(item_path)
                
                # 移除 zip 文件本身
                image_files = [f for f in image_files if not f.endswith('.zip')]
                
                if len(image_files) != 2:
                    all_results[patient_id] = {"error": "未找到正确数量的图片文件（需要2张）"}
                    continue
                
                # 使用前两张图片作为左右眼（顺序不重要）
                image_path_1 = image_files[0]
                image_path_2 = image_files[1]
                
                # 进行预测
                results = predict(model, image_path_1, image_path_2)
                
                # 处理预测结果
                if isinstance(results, dict) and 'prediction_results' in results:
                    all_results[patient_id] = {
                        "diseases": results["prediction_results"]["diseases"]
                    }
                else:
                    formatted_results = {"diseases": {}}
                    for result in results:
                        disease_name = result["class"]
                        prob_value = float(result["probability"])
                        predicted = bool(result["predicted"])
                        
                        formatted_results["diseases"][disease_name] = {
                            "probability": prob_value,
                            "status": 1 if predicted else 0
                        }
                    all_results[patient_id] = formatted_results
                    
            except Exception as e:
                all_results[patient_id] = {"error": str(e)}
                
            finally:
                # 清理临时文件
                if os.path.exists(patient_temp_dir):
                    for file in os.listdir(patient_temp_dir):
                        try:
                            os.remove(os.path.join(patient_temp_dir, file))
                        except:
                            pass
                    try:
                        os.rmdir(patient_temp_dir)
                    except:
                        pass
        
        return jsonify({"prediction_results": all_results})
        
    except Exception as e:
        import traceback
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("错误信息:", error_info)
        return jsonify(error_info), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)