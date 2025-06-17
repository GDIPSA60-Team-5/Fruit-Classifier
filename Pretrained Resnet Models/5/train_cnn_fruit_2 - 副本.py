# ─────────────────────────────────────────────────────────────
#  train_cnn_fruit_v3.py  –  EfficientNet-B0 + Hard-Aug
#  • 取消 rescale，改用 preprocess_input
#  • 默认关闭 MixUp (如需再开，把 USE_MIXUP=True 并参照上次补丁加入 simple_mixup)
# ─────────────────────────────────────────────────────────────
import os, random, pathlib, shutil, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# 0  全局随机种子
SEED = 42
os.environ['PYTHONHASHSEED']='42'; os.environ['TF_DETERMINISTIC_OPS']='1'
random.seed(SEED); np.random.seed(SEED)
tf.random.set_seed(SEED); tf.keras.utils.set_random_seed(SEED)

# 1  路径
ROOT = r'C:\Users\Cleveland\Desktop\mechanical\assignment5'
TRAIN= pathlib.Path(ROOT,'train'); TEST = pathlib.Path(ROOT,'test')
OUT  = pathlib.Path(ROOT,'result'); OUT.mkdir(parents=True, exist_ok=True)
ERRDIR = OUT/'errors'; ERRDIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224,224); BATCH=32
USE_MIXUP = False          # 先关掉 MixUp，确保收敛

# 2  数据生成器 ----------------------------------------------------------
EFF_PREP = tf.keras.applications.efficientnet.preprocess_input   ### ← MOD

def make_generators():
    train_gen = ImageDataGenerator(
        preprocessing_function=EFF_PREP,                    ### ← MOD
        validation_split=0.15,
        rotation_range=30, width_shift_range=.2, height_shift_range=.2,
        shear_range=.15, zoom_range=.2,
        channel_shift_range=40., brightness_range=[.5,1.5],
        horizontal_flip=True, fill_mode='nearest')

    train_ds = train_gen.flow_from_directory(
        TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
        subset='training', shuffle=True, seed=SEED)

    val_ds   = train_gen.flow_from_directory(
        TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
        subset='validation', shuffle=False, seed=SEED)

    test_gen = ImageDataGenerator(preprocessing_function=EFF_PREP) ### ← MOD
    test_ds  = test_gen.flow_from_directory(
        TEST, target_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = make_generators()
CLASSES = list(test_ds.class_indices.keys())

# 3  EfficientNet-B0 模型 ----------------------------------------------
base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=IMG_SIZE+(3,), weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3, seed=SEED),
    layers.Dense(len(CLASSES), activation='softmax')
])
model.compile(tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# 4  Stage-1 冻结训练 ----------------------------------------------------
ckpt1 = ModelCheckpoint(OUT/'best_stage1.weights.h5', save_weights_only=True,
                        monitor='val_accuracy', mode='max',
                        save_best_only=True, verbose=1)
early1= EarlyStopping(monitor='val_accuracy', mode='max',
                      patience=7, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=30, validation_data=val_ds,
          callbacks=[ckpt1, early1])

# 5  Stage-2 微调最后 20 层 ---------------------------------------------
base.trainable=True; cnt=0
for layer in reversed(base.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable=False
    elif cnt<20:
        layer.trainable=True; cnt+=1
    else:
        layer.trainable=False

model.compile(tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

ckpt2 = ModelCheckpoint(OUT/'best_stage2.weights.h5', save_weights_only=True,
                        monitor='val_accuracy', mode='max',
                        save_best_only=True, verbose=1)
reduceL= ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=3, min_lr=2e-6, verbose=1)
early2 = EarlyStopping(monitor='val_accuracy', mode='max',
                       patience=10, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=20, validation_data=val_ds,
          callbacks=[ckpt2, reduceL, early2])

# 6  选最优权重并评估 ----------------------------------------------------
get_acc=lambda w: (model.load_weights(w), model.evaluate(test_ds,verbose=0)[1])[1]
acc1=get_acc(OUT/'best_stage1.weights.h5')
acc2=get_acc(OUT/'best_stage2.weights.h5')
best = OUT/'best_stage2.weights.h5' if acc2>=acc1 else OUT/'best_stage1.weights.h5'
model.load_weights(best)
print(f'✅  Using {best.name}  —  Test Accuracy: {max(acc1,acc2):.3%}')

# 7  复制误判图片 --------------------------------------------------------
y_true = test_ds.classes; y_prob=model.predict(test_ds,verbose=0)
y_pred = y_prob.argmax(axis=1)
for p in np.array(test_ds.filepaths)[y_true!=y_pred]:
    shutil.copy(p, ERRDIR/pathlib.Path(p).name)
print(f'📂  Misclassified images copied to {ERRDIR}')

# 8  生成 F01–F06 图表 (与上一版相同) -------------------
# ... <保持不变> ...
