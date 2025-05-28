# 基于CBAM注意力机制的ResNet图像识别系统

## 项目简介

本项目实现了一个基于CBAM（Convolutional Block Attention Module）注意力机制的ResNet图像识别系统，使用LFW（Labeled Faces in the Wild）数据集进行人脸识别任务。项目对比了标准ResNet模型和添加CBAM注意力机制的ResNet模型在识别性能上的差异，以验证注意力机制对模型性能的影响。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- PIL

可以通过以下命令安装所需依赖：

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn seaborn pillow
```

## 项目结构

```
.
├── data/                   # 存放数据集和预处理后的数据
│   └── lfw-deepfunneled/   # LFW数据集
│   └── people.csv          # 人物信息CSV文件
├── models/                 # 存放模型定义
│   ├── resnet.py           # 标准ResNet模型
│   ├── cbam.py             # CBAM注意力机制模块
│   └── cbam_resnet.py      # 带CBAM的ResNet模型
├── scripts/                # 存放辅助脚本
│   ├── data_loader.py      # 数据加载和预处理
│   └── train.py            # 训练和评估函数
├── logs/                   # 存放训练日志和模型检查点
├── results/                # 存放实验结果和模型评估报告
├── main.py                 # 主程序
└── README.md               # 项目说明
```

## 使用方法

### 1. 准备数据

将LFW数据集放在`data/lfw-deepfunneled/lfw-deepfunneled`目录下，确保`data/people.csv`文件存在。

### 2. 运行程序

```bash
python main.py
```

程序将自动执行以下步骤：

1. 加载和预处理数据
2. 训练标准ResNet模型
3. 训练带CBAM注意力机制的ResNet模型
4. 评估两个模型的性能
5. 生成对比报告和可视化结果

### 3. 查看结果

训练完成后，可以在`results/`目录下查看以下结果：

- `resnet_history.png`：标准ResNet模型的训练历史曲线
- `cbam_resnet_history.png`：带CBAM的ResNet模型的训练历史曲线
- `model_comparison.png`：两个模型性能对比图
- `resnet_confusion_matrix.png`：标准ResNet模型的混淆矩阵
- `cbam_resnet_confusion_matrix.png`：带CBAM的ResNet模型的混淆矩阵
- `resnet_classification_report.txt`：标准ResNet模型的分类报告
- `cbam_resnet_classification_report.txt`：带CBAM的ResNet模型的分类报告
- `model_comparison_report.txt`：模型对比报告

## 模型说明

### ResNet模型

本项目使用ResNet-18作为基础模型，它是一种残差网络，通过跳跃连接解决了深度神经网络的梯度消失问题。

### CBAM注意力机制

CBAM（Convolutional Block Attention Module）是一种注意力机制，它包含两个子模块：

1. **通道注意力模块**：关注"什么"是有意义的特征
2. **空间注意力模块**：关注"哪里"有有意义的特征

CBAM通过顺序应用这两个注意力模块，有效地提高了模型的表示能力。

## 参数设置

- 图像大小：64x64像素
- 批量大小：16
- 训练轮数：100
- 学习率：0.001
- 权重衰减：0.0001
- 优化器：Adam
- 学习率调度器：ReduceLROnPlateau

## 防止过拟合的策略

本项目采用以下策略防止过拟合：

1. **数据增强**：随机水平翻转、随机旋转、颜色抖动
2. **权重衰减**：L2正则化
3. **Dropout**：在全连接层前应用
4. **学习率调度**：当验证损失不再下降时降低学习率

## 注意事项

- 训练过程可能需要较长时间，取决于硬件配置
- 如果显存不足，可以尝试减小批量大小或图像大小
- 可以通过修改`main.py`中的超参数来调整模型性能