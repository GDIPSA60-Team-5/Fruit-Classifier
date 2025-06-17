import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

# 设置数据路径
train_dir = r'C:\Users\Cleveland\Desktop\mechanical\assignment8\train'

# 数据增强和归一化
train_datagen = ImageDataGenerator(
    rescale=1./255,        # 归一化
    rotation_range=20,     # 旋转范围
    width_shift_range=0.2, # 水平移动
    height_shift_range=0.2, # 垂直移动
    shear_range=0.2,       # 剪切变换
    zoom_range=0.2,        # 缩放范围
    horizontal_flip=True,  # 随机水平翻转
    fill_mode='nearest'    # 填充模式
)

# 使用flow_from_directory从目录加载图片
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224), # 目标大小
    batch_size=32,
    class_mode='categorical' # 类别标签
)

# 构建不使用预训练权重的MobileNetV2模型
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))  # 去掉预训练
base_model.trainable = True  # 允许训练

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_generator.num_classes, activation='softmax') # 输出层
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 早期停止回调
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    callbacks=[early_stopping]
)

# 评估模型
model.evaluate(train_generator)

# 保存模型
model.save('fruit_mobilenetv2_no_pretrained.h5')

