import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import os
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


class EyeImageProcessor:
    """
    眼底图像处理类，提供多种眼科图像特定的预处理方法
    """
    def __init__(self, image_path=None, image=None):
        """
        初始化图像处理器
        
        参数:
            image_path (str): 图像路径
            image (PIL.Image/numpy.ndarray): 图像对象
        """
        if image_path is not None:
            self.load_image(image_path)
        elif image is not None:
            self.set_image(image)
        else:
            self.image = None
            self.image_cv = None
    
    def load_image(self, image_path):
        """
        加载图像
        
        参数:
            image_path (str): 图像路径
        """
        self.image = Image.open(image_path).convert('RGB')
        self.image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        return self
    
    def set_image(self, image):
        """
        设置图像
        
        参数:
            image (PIL.Image/numpy.ndarray): 图像对象
        """
        if isinstance(image, Image.Image):
            self.image = image
            self.image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            self.image_cv = image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 假设输入是BGR格式（OpenCV默认）
                self.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                self.image = Image.fromarray(image)
        return self
    
    def get_image(self, as_pil=True):
        """
        获取处理后的图像
        
        参数:
            as_pil (bool): 是否返回PIL图像对象
        
        返回:
            image (PIL.Image/numpy.ndarray): 图像对象
        """
        if as_pil:
            return self.image
        else:
            return self.image_cv
    
    def apply_clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        应用CLAHE（对比度受限的自适应直方图均衡化）
        
        参数:
            clip_limit (float): 对比度限制
            tile_grid_size (tuple): 网格大小
        """
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE到L通道
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # 合并通道
        limg = cv2.merge((cl, a, b))
        
        # 转换回BGR
        self.image_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        return self
    
    def apply_gamma_correction(self, gamma=1.2):
        """
        应用伽马校正
        
        参数:
            gamma (float): 伽马值
        """
        # 计算伽马校正查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        # 应用查找表
        self.image_cv = cv2.LUT(self.image_cv, table)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        return self
    
    def apply_sharpening(self, kernel_size=3, strength=1.0):
        """
        应用锐化
        
        参数:
            kernel_size (int): 核大小
            strength (float): 锐化强度
        """
        if kernel_size == 3:
            # 使用3x3锐化核
            kernel = np.array([[-strength, -strength, -strength],
                              [-strength, 1 + 8*strength, -strength],
                              [-strength, -strength, -strength]])
        else:
            # 使用拉普拉斯算子
            kernel = cv2.getGaussianKernel(kernel_size, 0)
            kernel = -strength * (kernel @ kernel.T)
            kernel[kernel_size//2, kernel_size//2] += 1.0
        
        # 应用滤波器
        self.image_cv = cv2.filter2D(self.image_cv, -1, kernel)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        return self
    
    def apply_denoising(self, h=10, template_window_size=7, search_window_size=21):
        """
        应用去噪
        
        参数:
            h (int): 滤波强度
            template_window_size (int): 模板窗口大小
            search_window_size (int): 搜索窗口大小
        """
        # 应用非局部均值去噪
        self.image_cv = cv2.fastNlMeansDenoisingColored(
            self.image_cv, None, h, h, template_window_size, search_window_size
        )
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        return self
    
    def enhance_vessels(self, sigma=2.0, threshold=10):
        """
        增强血管
        
        参数:
            sigma (float): 高斯滤波器的标准差
            threshold (int): 二值化阈值
        """
        # 转换为灰度图
        gray = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯滤波
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # 计算拉普拉斯算子
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # 归一化
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # 二值化
        _, binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 创建彩色掩码
        mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 将掩码与原图叠加
        alpha = 0.7
        self.image_cv = cv2.addWeighted(self.image_cv, 1.0, mask, alpha, 0)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        return self
    
    def detect_optic_disc(self, min_radius=50, max_radius=100):
        """
        检测视盘
        
        参数:
            min_radius (int): 最小半径
            max_radius (int): 最大半径
        
        返回:
            center (tuple): 视盘中心坐标
            radius (int): 视盘半径
        """
        # 转换为灰度图
        gray = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯滤波
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用霍夫圆变换检测圆
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles is not None:
            # 转换为整数坐标
            circles = np.round(circles[0, :]).astype("int")
            
            # 获取第一个圆（假设是视盘）
            x, y, r = circles[0]
            
            # 在图像上绘制圆
            cv2.circle(self.image_cv, (x, y), r, (0, 255, 0), 2)
            cv2.circle(self.image_cv, (x, y), 2, (0, 0, 255), 3)
            
            # 更新PIL图像
            self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
            
            return (x, y), r
        
        return None, None
    
    def detect_lesions(self, sensitivity=50):
        """
        检测病变区域
        
        参数:
            sensitivity (int): 灵敏度，值越小检测越敏感
        
        返回:
            contours (list): 病变区域轮廓
        """
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE到L通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 应用阈值分割
        _, thresh = cv2.threshold(cl, 200 - sensitivity, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        # 在图像上绘制轮廓
        cv2.drawContours(self.image_cv, contours, -1, (0, 0, 255), 2)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        
        return contours
    
    def crop_to_circle(self, padding=0):
        """
        将图像裁剪为圆形
        
        参数:
            padding (int): 边距
        """
        # 获取图像尺寸
        height, width = self.image_cv.shape[:2]
        
        # 创建圆形掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = min(center[0], center[1]) - padding
        cv2.circle(mask, center, radius, 255, -1)
        
        # 应用掩码
        result = cv2.bitwise_and(self.image_cv, self.image_cv, mask=mask)
        
        # 创建透明背景
        b, g, r = cv2.split(result)
        alpha = mask
        result_bgra = cv2.merge([b, g, r, alpha])
        
        # 更新图像
        self.image_cv = result
        self.image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        return self
    
    def adjust_brightness_contrast(self, brightness=1.0, contrast=1.0):
        """
        调整亮度和对比度
        
        参数:
            brightness (float): 亮度因子
            contrast (float): 对比度因子
        """
        # 使用PIL的ImageEnhance调整亮度和对比度
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(self.image)
            self.image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(self.image)
            self.image = enhancer.enhance(contrast)
        
        # 更新OpenCV图像
        self.image_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        
        return self
    
    def apply_color_balance(self):
        """
        应用颜色平衡
        """
        # 使用简单的白平衡算法
        result = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        # 调整LAB空间中的A和B通道
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        # 转换回BGR
        self.image_cv = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        
        return self
    
    def apply_histogram_equalization(self):
        """
        应用直方图均衡化
        """
        # 转换为YUV颜色空间
        yuv = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2YUV)
        
        # 对Y通道应用直方图均衡化
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # 转换回BGR
        self.image_cv = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 更新PIL图像
        self.image = Image.fromarray(cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB))
        
        return self
    
    def save_image(self, save_path):
        """
        保存图像
        
        参数:
            save_path (str): 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        self.image.save(save_path)
        
        return self
    
    def show_image(self, title="Processed Image"):
        """
        显示图像
        
        参数:
            title (str): 图像标题
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.image)
        plt.title(title)
        plt.axis('off')
        plt.show()
        
        return self
    
    def compare_with_original(self, original_image):
        """
        与原始图像进行比较
        
        参数:
            original_image (PIL.Image/numpy.ndarray): 原始图像
        """
        # 确保原始图像是PIL图像
        if isinstance(original_image, np.ndarray):
            original_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        # 创建画布
        plt.figure(figsize=(15, 8))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("原始图像")
        plt.axis('off')
        
        # 显示处理后的图像
        plt.subplot(1, 2, 2)
        plt.imshow(self.image)
        plt.title("处理后的图像")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return self


def process_batch(image_dir, output_dir, preprocessing_methods=None):
    """
    批量处理图像
    
    参数:
        image_dir (str): 图像目录
        output_dir (str): 输出目录
        preprocessing_methods (list): 预处理方法列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 处理每个图像
    for image_file in image_files:
        # 构建路径
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        # 创建处理器
        processor = EyeImageProcessor(image_path=image_path)
        
        # 应用预处理方法
        if preprocessing_methods:
            for method in preprocessing_methods:
                if method == 'clahe':
                    processor.apply_clahe()
                elif method == 'gamma':
                    processor.apply_gamma_correction()
                elif method == 'sharpen':
                    processor.apply_sharpening()
                elif method == 'denoise':
                    processor.apply_denoising()
                elif method == 'vessels':
                    processor.enhance_vessels()
                elif method == 'color_balance':
                    processor.apply_color_balance()
        
        # 保存处理后的图像
        processor.save_image(output_path)
        
        print(f"已处理: {image_file}")


if __name__ == "__main__":
    # 示例：处理单个图像
    image_path = "D:/Fc25_07/FcccccUestc/Training_data/0_left.jpg"
    output_path = "D:/Fc25_07/FcccccUestc/results/processed_0_left.jpg"
    
    # 创建处理器
    processor = EyeImageProcessor(image_path=image_path)
    
    # 应用预处理方法
    processor.apply_clahe().apply_sharpening().enhance_vessels()
    
    # 保存处理后的图像
    processor.save_image(output_path)
    
    # 显示对比
    original_image = Image.open(image_path).convert('RGB')
    processor.compare_with_original(original_image)
    
    # 示例：批量处理图像
    # process_batch(
    #     image_dir="D:/Fc25_07/FcccccUestc/Training_data",
    #     output_dir="D:/Fc25_07/FcccccUestc/results/processed",
    #     preprocessing_methods=['clahe', 'sharpen']
    # )
    
    print("处理完成!")
