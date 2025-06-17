# train_cnn_fruit_1.py
# 全部去除了权重文件保存和读取，保留两阶段训练和图表生成

import os, random, pathlib, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# 集群积累秒的seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED); np.random.seed(SEED)
tf.random.set_seed(SEED); tf.keras.utils.set_random_seed(SEED)

# 路径
ROOT   = r'C:\Users\Cleveland\Desktop\mechanical\assignment4'
TRAIN  = pathlib.Path(ROOT, 'train')
TEST   = pathlib.Path(ROOT,  'test')
OUT    = pathlib.Path(ROOT,  'result'); OUT.mkdir(exist_ok=True)

IMG_SIZE = (224, 224);  BATCH = 32

# 数据加载
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

# stage-1 冷冻基础组件
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

early1 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=6, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=25, validation_data=val_ds,
          callbacks=[early1])

# stage-2 微调后10层
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

reduceL = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    min_lr=3e-6, verbose=1)
early2 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=8, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=20, validation_data=val_ds,
          callbacks=[reduceL, early2])

# 最终测试
loss, acc = model.evaluate(test_ds, verbose=0)
print(f'✅  Final Test Accuracy: {acc:.3%}')

# 图表生成
print("⬆️ Saving figures to:", OUT)
y_true = test_ds.classes
y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# confusion matrix plot
def plot_confusion(cm, labels, norm, fname):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix' if norm else 'Confusion Matrix')
    for (i,j), v in np.ndenumerate(cm):
        txt = f"{v:0.2%}" if norm else str(v)
        ax.text(j, i, txt, ha='center', va='center', color='black')
    plt.tight_layout(); plt.savefig(OUT/fname, dpi=300); plt.close()

plot_confusion(cm, CLASSES, False, 'F01_confusion.png')
plot_confusion(cm_norm, CLASSES, True,  'F02_confusion_norm.png')

# label distribution
fig, ax = plt.subplots(1,2, figsize=(10,4))
counts = np.bincount(y_true, minlength=len(CLASSES))
ax[0].bar(CLASSES, counts, color='#1f77b4')
ax[0].set_ylabel('Images'); ax[0].set_title('Label Distribution')
ax[1].axis('off'); ax[1].text(0.5,0.5,'No bbox data',
                              ha='center', va='center')
plt.tight_layout(); plt.savefig(OUT/'F03_label_bbox.png', dpi=300)
plt.close()

# precision recall curve
probs_max  = y_prob.max(axis=1)
thresholds = np.linspace(0,1,101)
prec_list, rec_list, f1_list = [], [], []

for t in thresholds:
    mask = probs_max >= t
    if mask.sum()==0:
        prec, rec, f1 = 1.0, 0.0, 0.0
    else:
        rpt = classification_report(y_true[mask], y_pred[mask],
                                    output_dict=True, zero_division=0)
        prec = rpt['weighted avg']['precision']
        rec  = rpt['weighted avg']['recall']
        f1   = rpt['weighted avg']['f1-score']
    prec_list.append(prec); rec_list.append(rec); f1_list.append(f1)

def save_curve(x, y, xlabel, ylabel, title, fname):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, color='#1f77b4')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT/fname, dpi=300); plt.close()

save_curve(thresholds, f1_list,
           'Confidence threshold', 'Weighted F1',
           'F1 vs Confidence', 'F04_f1_confidence.png')

save_curve(thresholds, prec_list,
           'Confidence threshold', 'Weighted Precision',
           'Precision vs Confidence', 'F05_precision_confidence.png')

save_curve(rec_list, prec_list,
           'Recall', 'Precision',
           'Precision-Recall curve (macro-avg)', 'F06_precision_recall.png')
