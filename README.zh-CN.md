# 手写数字识别系统

[![English](https://img.shields.io/badge/English-README.md-blue)](README.md)

基于PyTorch和ResNet架构的实时手写数字识别系统，在MNIST数据集上达到99.2%的测试准确率。

## 🎯 项目概述

本项目实现了一个完整的手写数字识别系统，提供三种主要模式：
- **实时摄像头检测** - 实时网络摄像头数字识别（~60 FPS）
- **图像检测** - 使用高级预处理处理静态图像
- **模型训练** - 在MNIST数据集上训练ResNet并包含数据增强

## 🚀 主要特性

- **高精度**：99.69%训练准确率，99.20%测试准确率
- **实时性能**：~60 FPS摄像头检测，~200 FPS图像预处理
- **鲁棒预处理**：处理各种光照条件和图像质量
- **多种输入模式**：摄像头、静态图像、批量处理
- **GPU加速**：支持CUDA，可回退到CPU
- **教育价值**：完整的计算机视觉和深度学习实现

## 📋 环境要求

- **Python**：3.8+
- **平台**：Windows 11（已测试），Linux/macOS（应该可用）
- **IDE**：PyCharm（推荐）、VS Code或任何Python IDE

### 依赖项
```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

## 📁 项目结构

```
Hand_wrtten/
├── dataset/           # MNIST数据集（.idx3-ubyte格式）
├── logs/              # 训练好的模型权重（.pth文件）
├── real_img/          # 示例测试图像（JPG）
├── test_imgs/         # 额外测试图像（BMP）
├── __pycache__/       # Python字节码缓存
├── training_plot.png  # 训练可视化图表
├── .gitignore         # Git忽略规则
├── main.py            # 实时摄像头检测
├── main_pthoto.py     # 基于图像的检测
├── hand_wrtten_train.py # 模型训练
├── predict.py         # 神经网络预测函数
└── Pre_treatment.py   # 图像预处理工具
```

## 🔧 技术实现

### 神经网络架构
- **ResNet（残差网络）** 包含批量归一化
- **输入**：28×28灰度图像
- **输出**：10个类别（数字0-9）
- **架构**：Conv2d → BatchNorm → ReLU → MaxPool → 3个ResNet块 → GlobalAvgPool → FC

### 图像预处理流程
1. 灰度转换
2. 高斯模糊降噪
3. 自适应阈值处理（高斯C）
4. 形态学操作（闭运算）
5. 轮廓检测和裁剪
6. 居中和调整大小到28×28

### 数据集
- **MNIST数据集**：60,000张训练图像 + 10,000张测试图像
- **数据增强**：随机旋转、平移和缩放
- **格式**：28×28灰度图像

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/juzipaome/Hand_wrtten.git
cd Hand_wrtten
```

### 2. 安装依赖项
```bash
pip install -r requirements.txt
```

### 3. 选项A：使用预训练模型（推荐）
仓库在`logs/`文件夹中包含预训练模型权重。跳到第4步。

### 3. 选项B：训练自己的模型
```bash
python hand_wrtten_train.py
```
**注意**：如果需要，更新`hand_wrtten_train.py`中的数据集路径。

### 4. 用图像测试
```bash
python main_pthoto.py
```
**注意**：更新`main_pthoto.py`中的图像路径和`predict.py`中的模型路径。

### 5. 实时摄像头检测
```bash
python main.py
```
**注意**：如果你训练了自己的模型，更新`predict.py`中的模型路径。

## 📊 性能指标

| 指标 | 数值 |
|--------|--------|
| 训练准确率 | 99.69% |
| 测试准确率 | 99.20% |
| 实时FPS | ~60 FPS |
| 预处理FPS | ~200 FPS |
| 训练时间（RTX 2070） | ~1小时 |

## 🎯 使用示例

### 实时摄像头检测
```python
# 运行main.py进行实时网络摄像头检测
python main.py
```
- 显示带有数字识别叠加的实时摄像头馈送
- 按'q'退出
- 优化为~60 FPS性能

### 基于图像的检测
```python
# 运行main_pthoto.py进行静态图像处理
python main_pthoto.py
```
- 处理`real_img/`和`test_imgs/`文件夹中的图像
- 显示原始图像、预处理图像和预测结果
- 支持JPG和BMP格式

### 模型训练
```python
# 运行hand_wrtten_train.py从头开始训练
python hand_wrtten_train.py
```
- 在MNIST数据集上训练ResNet
- 包含数据增强和验证
- 将模型权重保存到`logs/`文件夹

## 🔍 配置

### 关键参数
- **模型路径**：在`predict.py`中更新（约第15行）
- **数据集路径**：在`hand_wrtten_train.py`中更新（约第20行）
- **图像路径**：在`main_pthoto.py`中更新（约第10行）
- **摄像头索引**：如果使用非默认摄像头，在`main.py`中更新

### 预处理参数
- 高斯模糊核大小：`(5, 5)`
- 自适应阈值块大小：`11`
- 形态学操作核：`(3, 3)`
- 最终图像大小：`28×28`

## 🛠️ 故障排除

### 常见问题
1. **摄像头不工作**：检查`main.py`中的摄像头索引
2. **找不到模型**：验证`predict.py`中的模型路径
3. **准确率低**：检查图像预处理质量
4. **性能慢**：启用CUDA GPU加速

### 性能提示
- 使用GPU加速获得更好性能
- 为你的特定用例调整预处理参数
- 确保摄像头检测有良好的光照条件
- 使用高对比度图像获得最佳结果

## 📚 教育资源

本项目非常适合：
- 计算机视觉课程项目
- 深度学习初学者
- PyTorch学习练习
- 实时检测应用
- 图像预处理技术

## 🤝 贡献

欢迎提交问题和增强请求！

## 📄 许可证

本项目是开源的，可在MIT许可证下使用。

## 🙏 致谢

- MNIST数据集创建者
- PyTorch社区
- OpenCV贡献者

---


**语言：** [![English](https://img.shields.io/badge/English-README.md-blue)](README.md) [![中文](https://img.shields.io/badge/中文-当前-red)](README.zh-CN.md)