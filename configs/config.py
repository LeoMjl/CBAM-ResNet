# 项目配置文件

# 数据集配置
DATASET = {
    'data_dir': 'data/lfw-deepfunneled/lfw-deepfunneled',  # 数据集目录
    'people_csv': 'data/people.csv',                      # 人物信息CSV文件
    'num_classes': 50,                                    # 使用前50个类别
    'min_images_per_class': 5,                            # 每个类别至少5张图像
    'image_size': 64,                                     # 图像大小 (64x64)
    'train_ratio': 0.8,                                   # 训练集比例
    'random_seed': 42                                     # 随机种子
}

# 数据加载配置
DATALOADER = {
    'batch_size': 16,                                     # 批量大小
    'num_workers': 4,                                     # 数据加载线程数
    'shuffle': True,                                      # 是否打乱数据
    'pin_memory': True                                    # 是否将数据加载到CUDA固定内存
}

# 数据增强配置
AUGMENTATION = {
    'horizontal_flip_prob': 0.5,                         # 水平翻转概率
    'rotation_degrees': 10,                               # 随机旋转角度范围
    'color_jitter': {                                     # 颜色抖动参数
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'normalize': {                                        # 归一化参数
        'mean': [0.485, 0.456, 0.406],                   # ImageNet均值
        'std': [0.229, 0.224, 0.225]                     # ImageNet标准差
    }
}

# 模型配置
MODEL = {
    'resnet': {
        'name': 'ResNet18',                               # 模型名称
        'pretrained': False,                              # 不使用预训练模型
        'dropout_rate': 0.5                               # Dropout比例
    },
    'cbam_resnet': {
        'name': 'CBAMResNet18',                           # 模型名称
        'pretrained': False,                              # 不使用预训练模型
        'dropout_rate': 0.5,                              # Dropout比例
        'reduction_ratio': 16                             # CBAM通道注意力降维比例
    }
}

# 训练配置
TRAINING = {
    'num_epochs': 100,                                   # 训练轮数
    'learning_rate': 0.001,                               # 初始学习率
    'weight_decay': 1e-4,                                 # 权重衰减（L2正则化）
    'optimizer': 'adam',                                  # 优化器类型
    'scheduler': {                                        # 学习率调度器
        'factor': 0.1,                                    # 学习率衰减因子
        'patience': 10,                                   # 容忍轮数
        'min_lr': 1e-6                                    # 最小学习率
    },
    'early_stopping_patience': 20                         # 早停容忍轮数
}

# 设备配置
DEVICE = {
    'use_cuda': True,                                     # 是否使用CUDA
    'cuda_device': 0                                      # CUDA设备ID
}

# 日志和保存配置
LOGGING = {
    'log_dir': 'logs',                                   # 日志目录
    'checkpoint_dir': 'logs/checkpoints',                 # 模型检查点目录
    'results_dir': 'results',                             # 结果目录
    'log_interval': 10                                    # 日志打印间隔（批次）
}