# author:Hurricane
# date:  2020/11/4
# E-mail:hurri_cane@qq.com
#
# 经过“优化1”和“优化3”修改后的版本
# 增加了数据归一化、数据增强 (Transforms) 和 DataLoader

import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2 as cv
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
from tqdm import tqdm

# --- 新增的导入 ---
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
# --- 新增导入结束 ---


# 训练集文件
train_images_idx3_ubyte_file = 'E:/File/Projects/Hand_wrtten/dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'E:/File/Projects/Hand_wrtten/dataset/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'E:/File/Projects/Hand_wrtten/dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'E:/File/Projects/Hand_wrtten/dataset/t10k-labels.idx1-ubyte'


# 读取数据部分 (这部分不变)
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'  
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张, 图片大小: %d*%d' % (num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  
    fmt_image = '>' + str(
        image_size) + 'B'  
    images = np.empty((num_images, 28, 28))
    for i in tqdm(range(num_images)):
        image = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols)).astype(np.uint8)
        images[i] = image
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('图片数量: %d张' % (num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in tqdm(range(num_images)):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

# --- 新增: CustomMNISTDataset 类 ---
# 这个类用于配合 DataLoader 和 Transforms
class CustomMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: 形状为 (N, 28, 28) 的 Numpy 数组, 像素值已归一化到 [0, 1]
        labels: 形状为 (N,) 的 Numpy 数组
        transform: 应用于图像的 torchvision.transforms
        """
        self.images = images.astype(np.float32) # 确保是 float32
        self.labels = labels.astype(np.int64)   # 确保是 int64 (CrossEntropyLoss 的要求)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # transforms.ToPILImage() 期望 (H, W) 或 (H, W, C)
            # 我们的数据是 (28, 28)，是 OK 的
            image = self.transform(image)
            
        return image, label
# --- CustomMNISTDataset 类结束 ---


# 构建网络部分 (这部分不变)
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

# --- 修改: evaluate_accuracy 函数 ---
# 修改为接受 DataLoader 作为输入，这更标准
def evaluate_accuracy(data_loader, net, device):
    """
    使用 DataLoader 评估模型在测试集上的准确率
    """
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        net.eval()  # 评估模式, 这会关闭dropout
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]
        net.train()  # 改回训练模式
    return acc_sum / n
# --- evaluate_accuracy 函数修改结束 ---


if __name__ == '__main__':
    # --- 主要修改区域开始 ---
    
    # 1. 加载数据并执行归一化 (优化 1)
    print("train:")
    # 归一化: 将像素值从 [0, 255] 缩放到 [0, 1]
    train_images_org = load_train_images().astype(np.float32) / 255.0
    train_labels_org = load_train_labels().astype(np.int64)
    
    print("test")
    # 归一化: 将像素值从 [0, 255] 缩放到 [0, 1]
    test_images_org = load_test_images().astype(np.float32) / 255.0
    test_labels_org = load_test_labels().astype(np.int64)

    # (可选) 不再需要查看原始图像，可以注释掉
    # for i in range(5):
    #     j = random.randint(0, 60000)
    #     print("now, show the number of image[{}]:".format(j), int(train_labels_org[j]))
    #     # 注意：显示归一化的图像
    #     img = cv.resize(train_images_org[j], (600, 600)) 
    #     cv.imshow("image", img)
    #     cv.waitKey(0)
    # cv.destroyAllWindows()
    # print('all done!')
    # print("*" * 50)

    # 2. 定义数据增强 (优化 3)
    # 训练集使用数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(), # 必须先转成 PIL Image 才能用后续的 transform
        # 增加随机旋转、平移和缩放
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor() # 会自动将 (H, W) 转为 (1, H, W) 并将像素值保持在 [0, 1]
    ])
    
    # 测试集不需要增强，只需要转为 Tensor
    test_transform = transforms.Compose([
        transforms.ToTensor() # 会自动将 (H, W) 转为 (1, H, W)
    ])

    # 3. 创建 Dataset 和 DataLoader (优化 3)
    train_dataset = CustomMNISTDataset(train_images_org, train_labels_org, transform=train_transform)
    # 只取前 1000 个测试数据（和您原来保持一致）
    test_dataset = CustomMNISTDataset(test_images_org[0:1000], test_labels_org[0:1000], transform=test_transform)

    batch_size = 128 # 可以适当调小 batch_size，例如 128 或 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. ResNet模型 (不变)
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))

    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 10)))

    # 测试网络 (不变)
    X = torch.rand((1, 1, 28, 28))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape:\t', X.shape)

    # 5. 训练 (循环部分修改)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, num_epochs = 0.001, 100 # Epoch 可以适当减少，因为增强后收敛可能变慢
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    net = net.to(device)

    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    
    # 确保 logs 目录存在
    if not os.path.exists('logs'):
        os.makedirs('logs')

    train_acc_plot = []
    test_acc_plot = []
    loss_plot = []
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        # --- 修改: 使用 DataLoader 循环 ---
        for X, y in tqdm(train_loader):
            # X 已经是 (batch, 1, 28, 28) 并且被增强了
            # y 已经是 (batch)
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # --- 循环修改结束 ---
            
        # --- 修改: 使用新的 evaluate_accuracy 函数 ---
        test_acc = evaluate_accuracy(test_loader, net, device)
        # --- 修改结束 ---
        
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
        torch.save(net.state_dict(), 'logs/Epoch%d-Loss%.4f-train_acc%.4f-test_acc%.4f.pth' % (
            (epoch + 1), train_l_sum / batch_count, train_acc_sum / n, test_acc))
        print("save successfully")

        test_acc_plot.append(test_acc)
        train_acc_plot.append(train_acc_sum / n)
        loss_plot.append(train_l_sum / batch_count)

    # 绘图 (不变)
    x = range(0, num_epochs) # 修正 x 的范围
    plt.plot(x, test_acc_plot, 'r', label='Test Acc')
    plt.plot(x, train_acc_plot, 'g', label='Train Acc')
    plt.plot(x, loss_plot, 'b', label='Loss')
    plt.legend()
    plt.savefig('training_plot.png') # 保存图像
    plt.show() # 显示图像
    
    print("*" * 50)
    # --- 主要修改区域结束 ---