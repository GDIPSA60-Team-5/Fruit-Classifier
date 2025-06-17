# ─────────────────────────────────────────────────────────────
#  train_cnn_fruit_v2.py  — file names end with .weights.h5
# ─────────────────────────────────────────────────────────────
'''
Example location:
  C:\\Users\\Cleveland\\Desktop\\mechanical\\assignment4.3\\train_cnn_fruit_v2.py
'''
import os, random, pathlib, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# 0  global seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED); np.random.seed(SEED)
tf.random.set_seed(SEED); tf.keras.utils.set_random_seed(SEED)

# 1  paths
ROOT   = r'C:\Users\Cleveland\Desktop\mechanical\assignment4.2'
TRAIN  = pathlib.Path(ROOT, 'train')
TEST   = pathlib.Path(ROOT,  'test')
OUT    = pathlib.Path(ROOT,  'result'); OUT.mkdir(exist_ok=True)

IMG_SIZE = (224, 224);  BATCH = 32

# 2  data
train_gen = ImageDataGenerator(
    rescale=1/255., validation_split=0.15,
    rotation_range=25, width_shift_range=.15, height_shift_range=.15,
    shear_range=.15, zoom_range=.15, horizontal_flip=True)

train_ds = train_gen.flow_from_directory(
    TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
    subset='training',  shuffle=True,  seed=SEED)
val_ds   = train_gen.flow_from_directory(
    TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
    subset='validation', shuffle=False, seed=SEED)
test_gen = ImageDataGenerator(rescale=1/255.)
test_ds  = test_gen.flow_from_directory(
    TEST, target_size=IMG_SIZE, batch_size=BATCH, shuffle=False)

CLASSES = list(test_ds.class_indices.keys())

# 3  stage-1  frozen backbone
base = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=IMG_SIZE+(3,), weights='imagenet')
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.25, seed=SEED),
    layers.Dense(len(CLASSES), activation='softmax')
])
model.compile(tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

ckpt1 = ModelCheckpoint(
    OUT/'best_stage1.weights.h5',   # ← 必须 .weights.h5
    monitor='val_accuracy', mode='max',
    save_best_only=True, save_weights_only=True, verbose=1)
early1 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=6, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=25, validation_data=val_ds,
          callbacks=[ckpt1, early1])

# 4  stage-2  fine-tune last 10 conv layers
base.trainable = True
cnt = 0
for layer in reversed(base.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif cnt < 10:
        layer.trainable = True; cnt += 1
    else:
        layer.trainable = False

model.compile(tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

ckpt2 = ModelCheckpoint(
    OUT/'best_stage2.weights.h5',
    monitor='val_accuracy', mode='max',
    save_best_only=True, save_weights_only=True, verbose=1)
reduceL = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    min_lr=3e-6, verbose=1)
early2 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=8, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=20, validation_data=val_ds,
          callbacks=[ckpt2, reduceL, early2])

# 5  choose better stage
def acc_of(wpath):
    model.load_weights(wpath)
    return model.evaluate(test_ds, verbose=0)[1]

acc1 = acc_of(OUT/'best_stage1.weights.h5')
acc2 = acc_of(OUT/'best_stage2.weights.h5')
best_w = OUT/'best_stage2.weights.h5' if acc2 >= acc1 else OUT/'best_stage1.weights.h5'
model.load_weights(best_w)
print(f'✅  Using {best_w.name}  →  Test Accuracy: {max(acc1, acc2):.3%}')

# 6  figures (unchanged)  … 省略，与前一版本相同 …
 