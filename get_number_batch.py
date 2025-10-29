# author:Hurricane
# date:  2020/11/5
# E-mail:hurri_cane@qq.com
#
# 版本更新：将预测结果显示在原始图片上

import cv2 as cv
import numpy as np
import os
import torch
# 从 predict.py 导入 get_net 和 predict 函数
from predict import get_net, predict

def process_real_image_for_mnist(img_path):
    """
    (这是新的预处理函数 - V2)
    加载一张用户手写的“真实”图片（背景可能不均匀，如您新上传的图片），
    将其处理成接近 MNIST 格式的 28x28 图像（黑底白字）。
    """
    img = cv.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图片 {img_path}")
        return None

    # 1. 转换为灰度图 (不变)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # --- 关键修改开始 ---
    
    # 2. (新增) 高斯模糊
    #    在阈值化之前进行模糊处理，可以去除高频噪点，使自适应阈值更稳定
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)

    # 3. (修改) 使用自适应阈值
    #    替换掉之前的全局 Otsu 阈值
    #    cv.ADAPTIVE_THRESH_GAUSSIAN_C: 使用高斯加权平均值计算局部阈值，效果更好
    #    cv.THRESH_BINARY_INV: 反转阈值，因为我们是浅色背景、深色数字 -> 变为黑底白字
    #    blockSize (11): 计算阈值的邻域大小（必须是奇数）
    #    C (2): 从均值或加权均值中减去的一个常数，用于微调
    #    (使用您在文件中调整后的参数)
    img_thresh = cv.adaptiveThreshold(
        img_blur, 
        255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 
        21, 
        5
    )
    
    # 4. (可选但推荐) 形态学操作
    #    (使用您在文件中调整后的参数)
    kernel = np.ones((3, 3), np.uint8)
    img_cleaned = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel, iterations=3)
    
    # --- 关键修改结束 ---

    # 5. 查找轮廓 (现在在 'img_cleaned' 上操作)
    contours, _ = cv.findContours(img_cleaned.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"未在图片 {img_path} 中找到轮廓")
        return None

    # 6. 找到最大的轮廓（即数字）
    try:
        largest_contour = max(contours, key=cv.contourArea)
    except ValueError:
        print(f"在 {img_path} 中未找到有效轮廓")
        return None
        
    x, y, w, h = cv.boundingRect(largest_contour)

    # 7. 裁剪出数字 (从清理后的图像 'img_cleaned' 中裁剪)
    img_cropped = img_cleaned[y:y+h, x:x+w]

    # 8. 调整大小并居中 (这部分逻辑不变)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    target_dim = 20 
    
    if w > h:
        new_w = target_dim
        new_h = int(h * (target_dim / w))
    else:
        new_h = target_dim
        new_w = int(w * (target_dim / h))

    if new_w < 1: new_w = 1
    if new_h < 1: new_h = 1
        
    img_resized = cv.resize(img_cropped, (new_w, new_h), interpolation=cv.INTER_AREA)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    return canvas

# --- 主程序开始 ---
if __name__ == '__main__':
    
    # 1. 加载您训练好的模型
    print("正在加载模型...")
    net = get_net()
    print("模型加载完毕。")

    # 2. 指定您手写图片的路径
    orig_path = r"real_img" 
    
    if not os.path.exists(orig_path):
        print(f"错误: 找不到路径 {orig_path}")
    else:
        img_list = os.listdir(orig_path)
        
        for img_name in img_list:
            if not img_name.lower().endswith(('.jpg', '.png', '.bmp')):
                continue

            img_path = os.path.join(orig_path, img_name)
            
            # 3. (关键) 使用新的 V2 预处理函数
            processed_img = process_real_image_for_mnist(img_path)
            
            if processed_img is not None:
                # 4. 传入 predict 函数
                result_tensor = predict(processed_img, net)
                
                # 5. 获取预测结果
                predicted_label = torch.argmax(result_tensor, dim=1).item()
                
                print(f"图片: {img_name}, 预测结果: {predicted_label}")

                # --- 关键修改：将结果绘制到图片上 ---
                
                # 读取原始图像用于显示
                orig_display = cv.imread(img_path)
                # 缩放图像以便显示
                orig_resized = cv.resize(orig_display, (300, 300))
                
                # 准备要绘制的文本
                text = f"Prediction: {predicted_label}"
                
                # 设置文本参数
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 0, 255) # 红色 (BGR)
                thickness = 2
                position = (10, 30) # 左上角坐标 (x, y)
                
                # 将文本绘制到缩放后的图像上
                cv.putText(orig_resized, text, position, font, font_scale, color, thickness)
                
                # 显示处理后的 28x28 图像 (用于调试)
                cv.imshow("Processed 28x28 (Target for MNIST)", cv.resize(processed_img, (300, 300), interpolation=cv.INTER_NEAREST))
                
                # 显示带有预测结果的原始图像
                cv.imshow("Original Image with Prediction", orig_resized)
                
                # --- 修改结束 ---

                print("按任意键查看下一张...")
                cv.waitKey(0)

        cv.destroyAllWindows()