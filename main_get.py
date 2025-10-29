# 文件名: realtime_predict.py
# 描述: 实时摄像头手写数字识别

import cv2 as cv
import numpy as np
import torch
import time

# 确保您可以从 predict.py 导入 get_net 和 predict
try:
    from predict import get_net, predict
except ImportError:
    print("错误: 无法导入 'predict.py'。")
    print("请确保 'realtime_predict.py' 和 'predict.py' 在同一个文件夹中。")
    exit()

def process_frame_for_mnist(frame):
    """
    (这是新的预处理函数 - V-Realtime)
    接收一个 NumPy 数组（视频帧）而不是文件路径。
    这是从 get_number_batch.py 复制并修改而来的，
    使用了您调优后的参数 (blockSize=21, C=5, iterations=3)。
    """
    if frame is None:
        return None

    # 1. 转换为灰度图 (我们假设输入 frame 已经是灰度图，如果不是，先转换)
    if len(frame.shape) > 2 and frame.shape[2] == 3:
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        img_gray = frame
    
    # --- 关键预处理逻辑 (来自 get_number_batch.py) ---
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    
    img_thresh = cv.adaptiveThreshold(
        img_blur, 
        255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 
        21, # 您调优的参数
        5   # 您调优的参数
    )
    
    kernel = np.ones((3, 3), np.uint8)
    img_cleaned = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel, iterations=3) # 您调优的参数
    
    # --- 预处理逻辑结束 ---

    # 5. 查找轮廓
    contours, _ = cv.findContours(img_cleaned.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # 没有找到轮廓

    # 6. 找到最大的轮廓
    try:
        largest_contour = max(contours, key=cv.contourArea)
    except ValueError:
        return None
        
    x, y, w, h = cv.boundingRect(largest_contour)

    # 7. 裁剪出数字
    img_cropped = img_cleaned[y:y+h, x:x+w]

    # 8. 调整大小并居中 (与 get_number_batch.py 保持一致)
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
    
    # 1. 加载模型
    print("正在加载模型...")
    # 注意：您的代码片段是 pt.get_net()，我猜您指的是 predict.get_net()
    # 如果您的文件名确实是 pt.py，请将下面的 'predict' 改为 'pt'
    net = get_net()
    print("模型加载完毕。")

    # 2. 初始化摄像头
    # capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    # CAP_DSHOW 在某些系统上更快，但 0 通常更通用
    capture = cv.VideoCapture(1,cv.CAP_DSHOW)
    # 3. 设置分辨率 (如您所愿)
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    # 注意：1920x1080 可能会导致帧率很低，推荐使用 1280x720
    capture.set(3, CAM_WIDTH)
    capture.set(4, CAM_HEIGHT)
    
    time.sleep(1) # 等待摄像头稳定
    
    # 4. 定义识别区域 (ROI)
    # 我们在屏幕中央定义一个 400x400 的方框
    roi_size = 400
    w_start = int((CAM_WIDTH - roi_size) / 2)
    h_start = int((CAM_HEIGHT - roi_size) / 2)
    w_end = w_start + roi_size
    h_end = h_start + roi_size

    prediction_text = "N/A" # 初始预测文本
    last_predict_time = time.time()

    print("摄像头启动... 按 'q' 退出。")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("错误: 无法读取摄像头帧")
            break
            
        # 翻转图像 (镜像模式)，这样更直观
     
        
        # 复制一份用于显示
        display_frame = frame.copy()
        
        # 提取识别区域 (ROI)
        roi = frame[h_start:h_end, w_start:w_end]
        
        # --- (关键) 每 0.25 秒进行一次识别 ---
        # 实时识别非常消耗资源，我们不需要每帧都识别
        current_time = time.time()
        if current_time - last_predict_time > 0.25: # 每秒识别 4 次
            last_predict_time = current_time
            
            # 5. 预处理 ROI
            processed_img = process_frame_for_mnist(roi)
            
            if processed_img is not None:
                # 6. 预测
                try:
                    result_tensor = predict(processed_img, net)
                    predicted_label = torch.argmax(result_tensor, dim=1).item()
                    prediction_text = str(predicted_label)
                    
                    # (可选) 显示处理后的 28x28 图像，用于调试
                    cv.imshow("Processed 28x28", cv.resize(processed_img, (200, 200), interpolation=cv.INTER_NEAREST))
                
                except Exception as e:
                    print(f"预测时出错: {e}")
                    prediction_text = "Error"
            else:
                # 如果没在 ROI 中找到轮廓
                prediction_text = "..."

        # 7. 绘制界面
        # 绘制 ROI 方框 (绿色)
        cv.rectangle(display_frame, (w_start, h_start), (w_end, h_end), (0, 255, 0), 2)
        
        # 绘制预测结果
        cv.putText(display_frame, f"Prediction: {prediction_text}", 
                   (w_start, h_start - 10),  # 文本放在方框上方
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2) # 红色

        # 8. 显示图像
        cv.imshow("Live Handwritting Recognition (Press 'q' to quit)", display_frame)

        # 9. 退出
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 10. 清理
    print("退出...")
    capture.release()
    cv.destroyAllWindows()