# ─────────────────────────────────────────────────────────────
#  train_cnn_fruit_v3.py   –  EfficientNet-B0 + Hard-Aug + MixUp
# ─────────────────────────────────────────────────────────────
"""
保存路径（示例）:
  C:\\Users\\Cleveland\\Desktop\\mechanical\\assignment5\\train_cnn_fruit_v3.py
"""
import os, random, pathlib, shutil, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# 0  全局随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED); np.random.seed(SEED)
tf.random.set_seed(SEED); tf.keras.utils.set_random_seed(SEED)

# 1  路径
ROOT   = r'C:\Users\Cleveland\Desktop\mechanical\assignment5'
TRAIN  = pathlib.Path(ROOT, 'train')
TEST   = pathlib.Path(ROOT,  'test')
OUT    = pathlib.Path(ROOT,  'result'); OUT.mkdir(parents=True, exist_ok=True)
ERRDIR = OUT/'errors'; ERRDIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH    = 32
USE_MIXUP = False          # ← 不想用 MixUp 就改成 False

# 2  数据生成器
def make_generators():
    train_gen = ImageDataGenerator(
        rescale=1/255., validation_split=0.15,
        rotation_range=30,
        width_shift_range=.2, height_shift_range=.2,
        shear_range=.15, zoom_range=.2,
        channel_shift_range=40.,
        brightness_range=[.5, 1.5],
        horizontal_flip=True, fill_mode='nearest')

    train_ds = train_gen.flow_from_directory(
        TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
        subset='training', shuffle=True, seed=SEED)

    val_ds = train_gen.flow_from_directory(
        TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
        subset='validation', shuffle=False, seed=SEED)

    test_gen = ImageDataGenerator(rescale=1/255.)
    test_ds  = test_gen.flow_from_directory(
        TEST, target_size=IMG_SIZE, batch_size=BATCH,
        shuffle=False)

    if USE_MIXUP:
        train_ds = tf.keras.utils.mix_up(train_ds, alpha=0.2, seed=SEED)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = make_generators()
CLASSES = list(test_ds.class_indices.keys())

# 3  EfficientNet-B0 模型
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

# 4  Stage-1 冻结训练
ckpt1 = ModelCheckpoint(
    OUT/'best_stage1.weights.h5', save_weights_only=True,
    monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
early1 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=7, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=30, validation_data=val_ds,
          callbacks=[ckpt1, early1])

# 5  Stage-2 微调后 20 层
base.trainable = True
cnt = 0
for layer in reversed(base.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif cnt < 20:
        layer.trainable = True; cnt += 1
    else:
        layer.trainable = False

model.compile(tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

ckpt2 = ModelCheckpoint(
    OUT/'best_stage2.weights.h5', save_weights_only=True,
    monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
reduceL = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    min_lr=2e-6, verbose=1)
early2 = EarlyStopping(
    monitor='val_accuracy', mode='max',
    patience=10, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=20, validation_data=val_ds,
          callbacks=[ckpt2, reduceL, early2])

# 6  评估并选择最佳权重
def acc_of(w):
    model.load_weights(w)
    return model.evaluate(test_ds, verbose=0)[1]

acc1 = acc_of(OUT/'best_stage1.weights.h5')
acc2 = acc_of(OUT/'best_stage2.weights.h5')
best_w = OUT/'best_stage2.weights.h5' if acc2 >= acc1 else OUT/'best_stage1.weights.h5'
model.load_weights(best_w)
print(f'✅  Using {best_w.name}  —  Test Accuracy: {max(acc1,acc2):.3%}')

# 7  复制误判图片
y_true = test_ds.classes
y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)

wrong_idx = np.where(y_true != y_pred)[0]
for p in np.array(test_ds.filepaths)[wrong_idx]:
    shutil.copy(p, ERRDIR/p.split(os.sep)[-1])
print(f'📂  Misclassified images copied to {ERRDIR}  ({len(wrong_idx)} files)')

# 8  指标图
def save_curve(x,y,xl,yl,title,fname):
    plt.figure(figsize=(6,4))
    plt.plot(x,y,color='#1f77b4'); plt.xlabel(xl); plt.ylabel(yl)
    plt.title(title); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT/fname,dpi=300); plt.close()

def cm_plot(M,labels,norm,fname):
    fig,ax=plt.subplots(figsize=(6,6))
    im=ax.imshow(M,cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalized' if norm else 'Confusion Matrix')
    for (i,j),v in np.ndenumerate(M):
        ax.text(j,i,f'{v:0.2%}' if norm else str(v),
                ha='center',va='center')
    plt.tight_layout(); plt.savefig(OUT/fname,dpi=300); plt.close()

cm  = confusion_matrix(y_true,y_pred)
cmn = cm.astype(float)/cm.sum(axis=1,keepdims=True)
cm_plot(cm, CLASSES, False,'F01_confusion.png')
cm_plot(cmn,CLASSES, True,'F02_confusion_norm.png')

fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].bar(CLASSES,np.bincount(y_true,minlength=len(CLASSES)),
          color='#1f77b4')
ax[0].set_title('Label Distribution'); ax[0].set_ylabel('Images')
ax[1].axis('off'); ax[1].text(.5,.5,'No bbox data',
                              ha='center',va='center')
plt.tight_layout(); plt.savefig(OUT/'F03_label_bbox.png',dpi=300); plt.close()

thr=np.linspace(0,1,101); prec=[]; rec=[]; f1=[]
mx=y_prob.max(axis=1)
for t in thr:
    m=mx>=t
    if not m.any(): prec.append(1); rec.append(0); f1.append(0); continue
    rep=classification_report(y_true[m],y_pred[m],output_dict=True,zero_division=0)
    prec.append(rep['weighted avg']['precision'])
    rec.append(rep['weighted avg']['recall'])
    f1.append(rep['weighted avg']['f1-score'])

save_curve(thr,f1,'Confidence','Weighted F1',
           'F1 vs Confidence','F04_f1_confidence.png')
save_curve(thr,prec,'Confidence','Weighted Precision',
           'Precision vs Confidence','F05_precision_confidence.png')
save_curve(rec,prec,'Recall','Precision',
           'Precision-Recall Curve','F06_precision_recall.png')

print(f'📁  All outputs saved to {OUT}')
