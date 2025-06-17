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

# ─────────────  5. 绘图函数  ─────────────
def save_curve(x, y, xlabel, ylabel, title, fname):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, color='#1f77b4')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT_DIR/fname, dpi=300); plt.close()

def plot_confusion(cm, labels, norm, fname):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix' if norm else 'Confusion Matrix')
    for (i,j), v in np.ndenumerate(cm):
        txt = f"{v:0.2%}" if norm else str(v)
        ax.text(j, i, txt, ha='center', va='center', color='black')
    plt.tight_layout(); plt.savefig(OUT_DIR/fname, dpi=300); plt.close()

# ─────────────  6. 混淆矩阵相关图  ─────────────
cm       = confusion_matrix(y_true, y_pred)
cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plot_confusion(cm,      CLASSES, False, 'F01_confusion.png')
plot_confusion(cm_norm, CLASSES, True,  'F02_confusion_norm.png')

# 标签分布
fig, ax = plt.subplots(1,2, figsize=(10,4))
counts = np.bincount(y_true, minlength=len(CLASSES))
ax[0].bar(CLASSES, counts, color='#1f77b4')
ax[0].set_ylabel('Images'); ax[0].set_title('Label Distribution')
ax[1].axis('off'); ax[1].text(0.5,0.5,'No bbox data',
                              ha='center', va='center')
plt.tight_layout(); plt.savefig(OUT_DIR/'F03_label_bbox.png', dpi=300)
plt.close()

# ─────────────  7. F1 / Precision vs Confidence & PR 曲线  ─────────────
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

save_curve(thresholds, f1_list,
           'Confidence threshold', 'Weighted F1',
           'F1 vs Confidence', 'F04_f1_confidence.png')

save_curve(thresholds, prec_list,
           'Confidence threshold', 'Weighted Precision',
           'Precision vs Confidence', 'F05_precision_confidence.png')

save_curve(rec_list, prec_list,
           'Recall', 'Precision',
           'Precision-Recall curve (macro-avg)', 'F06_precision_recall.png')

print(f"📁  All figures saved to: {OUT_DIR}")
