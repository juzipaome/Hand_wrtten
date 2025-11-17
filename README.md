# Handwritten Digit Recognition System

[![中文](https://img.shields.io/badge/中文-README--zh--CN.md-red)](README.zh-CN.md)

Thanks to the original author for the support of the code files.

A real-time handwritten digit recognition system using PyTorch and ResNet architecture, achieving 99.2% test accuracy on the MNIST dataset.

## 🎯 Project Overview

This project implements a complete handwritten digit recognition system with three main modes:
- **Real-time camera detection** - Live webcam digit recognition (~60 FPS)
- **Image-based detection** - Process static images with advanced preprocessing
- **Model training** - Train ResNet on MNIST with data augmentation

## 🚀 Key Features

- **High Accuracy**: 99.69% training accuracy, 99.20% test accuracy
- **Real-time Performance**: ~60 FPS camera detection, ~200 FPS image preprocessing
- **Robust Preprocessing**: Handles various lighting conditions and image quality
- **Multiple Input Modes**: Camera, static images, batch processing
- **GPU Acceleration**: CUDA support with CPU fallback
- **Educational Value**: Complete implementation for learning computer vision and deep learning

## 📋 Requirements

- **Python**: 3.8+
- **Platform**: Windows 10 (tested), Linux/macOS (should work)
- **IDE**: PyCharm (recommended), VS Code, or any Python IDE

### Dependencies
```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

## 📁 Project Structure

```
Hand_wrtten/
├── dataset/           # MNIST dataset (.idx3-ubyte format)
├── logs/              # Trained model weights (.pth files)
├── real_img/          # Sample test images (JPG)
├── test_imgs/         # Additional test images (BMP)
├── __pycache__/       # Python bytecode cache
├── training_plot.png  # Training visualization
├── .gitignore         # Git ignore rules
├── main.py            # Real-time camera detection
├── main_pthoto.py     # Image-based detection
├── hand_wrtten_train.py # Model training
├── predict.py         # Neural network prediction functions
└── Pre_treatment.py   # Image preprocessing utilities
```

## 🔧 Technical Implementation

### Neural Network Architecture
- **ResNet (Residual Network)** with batch normalization
- **Input**: 28×28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Architecture**: Conv2d → BatchNorm → ReLU → MaxPool → 3 ResNet blocks → GlobalAvgPool → FC

### Image Preprocessing Pipeline
1. Grayscale conversion
2. Gaussian blur for noise reduction
3. Adaptive thresholding (Gaussian C)
4. Morphological operations (closing)
5. Contour detection and cropping
6. Centering and resizing to 28×28

### Dataset
- **MNIST dataset**: 60,000 training + 10,000 test images
- **Data augmentation**: Random rotation, translation, and scaling
- **Format**: 28×28 grayscale images

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/juzipaome/Hand_wrtten.git
cd Hand_wrtten
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Option A: Use Pre-trained Model (Recommended)
The repository includes pre-trained model weights in the `logs/` folder. Skip to step 4.

### 3. Option B: Train Your Own Model
```bash
python hand_wrtten_train.py
```
**Note**: Update the dataset path in `hand_wrtten_train.py` if needed.

### 4. Test with Images
```bash
python main_pthoto.py
```
**Note**: Update image paths in `main_pthoto.py` and model path in `predict.py`.

### 5. Real-time Camera Detection
```bash
python main.py
```
**Note**: Update model path in `predict.py` if you trained your own model.

## 📊 Performance Metrics

| Metric | Value |
|--------|--------|
| Training Accuracy | 99.69% |
| Test Accuracy | 99.20% |
| Real-time FPS | ~60 FPS |
| Preprocessing FPS | ~200 FPS |
| Training Time (RTX 2070) | ~1 hour |

## 🎯 Usage Examples

### Real-time Camera Detection
```python
# Run main.py for live webcam detection
python main.py
```
- Shows live camera feed with digit recognition overlay
- Press 'q' to quit
- Optimized for ~60 FPS performance

### Image-based Detection
```python
# Run main_pthoto.py for static image processing
python main_pthoto.py
```
- Processes images in `real_img/` and `test_imgs/` folders
- Displays original image, preprocessed image, and prediction
- Supports JPG and BMP formats

### Model Training
```python
# Run hand_wrtten_train.py to train from scratch
python hand_wrtten_train.py
```
- Trains ResNet on MNIST dataset
- Includes data augmentation and validation
- Saves model weights to `logs/` folder

## 🔍 Configuration

### Key Parameters
- **Model Path**: Update in `predict.py` (line ~15)
- **Dataset Path**: Update in `hand_wrtten_train.py` (line ~20)
- **Image Paths**: Update in `main_pthoto.py` (line ~10)
- **Camera Index**: Update in `main.py` if using non-default camera

### Preprocessing Parameters
- Gaussian blur kernel size: `(5, 5)`
- Adaptive threshold block size: `11`
- Morphological operation kernel: `(3, 3)`
- Final image size: `28×28`

## 🛠️ Troubleshooting

### Common Issues
1. **Camera not working**: Check camera index in `main.py`
2. **Model not found**: Verify model path in `predict.py`
3. **Low accuracy**: Check image preprocessing quality
4. **Slow performance**: Enable GPU acceleration with CUDA

### Performance Tips
- Use GPU acceleration for better performance
- Adjust preprocessing parameters for your specific use case
- Ensure good lighting conditions for camera detection
- Use high-contrast images for best results

## 📚 Educational Resources

This project is ideal for:
- Computer vision course projects
- Deep learning beginners
- PyTorch learning exercises
- Real-time detection applications
- Image preprocessing techniques

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset creators
- PyTorch community
- OpenCV contributors


---

**Languages:** [![English](https://img.shields.io/badge/English-Current-blue)](README.md) [![中文](https://img.shields.io/badge/中文-README--zh--CN.md-red)](README.zh-CN.md)
