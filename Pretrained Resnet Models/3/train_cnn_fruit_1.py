# ─────────────────────────────────────────────────────────────
#  train_cnn_fruit.py    (place in assignment\)
# ─────────────────────────────────────────────────────────────
import os, random, pathlib, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# ───────────── 0. GLOBAL SEED  ─────────────
SEED = 42
os.environ["PYTHONHASHSEED"]      = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED);  np.random.seed(SEED)
tf.random.set_seed(SEED); tf.keras.utils.set_random_seed(SEED)

# ───────────── 1. PATHS & PARAMS  ─────────────
ROOT   = r"C:\Users\Cleveland\Desktop\mechanical\assignment3"
TRAIN  = pathlib.Path(ROOT, "train")   # train\apple …
TEST   = pathlib.Path(ROOT, "test")    # test\apple …
OUT    = pathlib.Path(ROOT, "result")
OUT.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS   = 50               # EarlyStopping 会提前结束

# ───────────── 2. DATA  ─────────────
train_gen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=0.15,
    rotation_range=25, width_shift_range=.1, height_shift_range=.1,
    shear_range=.1, zoom_range=.1, horizontal_flip=True)

train_ds = train_gen.flow_from_directory(
    TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
    subset="training",  shuffle=True,  seed=SEED)

val_ds   = train_gen.flow_from_directory(
    TRAIN, target_size=IMG_SIZE, batch_size=BATCH,
    subset="validation", shuffle=False, seed=SEED)

test_gen = ImageDataGenerator(rescale=1/255.)
test_ds  = test_gen.flow_from_directory(
    TEST, target_size=IMG_SIZE, batch_size=BATCH,
    shuffle=False)

CLASSES = list(test_ds.class_indices.keys())   # ['apple', 'banana', …]

# ───────────── 3. MODEL  ─────────────
base = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=IMG_SIZE+(3,), weights="imagenet")
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.25, seed=SEED),
    layers.Dense(len(CLASSES), activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

# ───────────── 4. TRAIN  ─────────────
ckpt = ModelCheckpoint(
        OUT/"best.h5", monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1)
early = EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=5, restore_best_weights=True, verbose=1)

model.fit(train_ds, epochs=EPOCHS,
          validation_data=val_ds,
          callbacks=[ckpt, early])

# ───────────── 5. *RESTORE* BEST WEIGHTS & TEST  ─────────────
model.load_weights(OUT/"best.h5")
print("✅  Restored best validation weights.")

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"✅  Final Test Accuracy: {test_acc:.3%}")

# predictions for plots
y_true = test_ds.classes
y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)

# ───────────── 6. PLOT HELPERS  ─────────────
def save_curve(x, y, xl, yl, title, fname):
    plt.figure(figsize=(6,4))
    plt.plot(x, y, color="#1f77b4")
    plt.xlabel(xl); plt.ylabel(yl); plt.title(title)
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT/fname, dpi=300); plt.close()

def plot_cm(mat, labels, norm, fname):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix" if norm else "Confusion Matrix")
    for (i,j), v in np.ndenumerate(mat):
        txt = f"{v:0.2%}" if norm else str(v)
        ax.text(j, i, txt, ha="center", va="center")
    plt.tight_layout(); plt.savefig(OUT/fname, dpi=300); plt.close()

# ───────────── 7. CONFUSION MATRICES  ─────────────
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plot_cm(cm,      CLASSES, False, "F01_confusion.png")
plot_cm(cm_norm, CLASSES, True,  "F02_confusion_norm.png")

# ───────────── 8. LABEL DISTRIBUTION  ─────────────
fig, ax = plt.subplots(1,2, figsize=(10,4))
counts = np.bincount(y_true, minlength=len(CLASSES))
ax[0].bar(CLASSES, counts, color="#1f77b4")
ax[0].set_title("Label Distribution"); ax[0].set_ylabel("Images")
ax[1].axis("off"); ax[1].text(0.5,0.5,"No bbox data",
                              ha="center", va="center")
plt.tight_layout(); plt.savefig(OUT/"F03_label_bbox.png", dpi=300); plt.close()

# ───────────── 9. F1 / PRECISION vs CONFIDENCE + PR CURVE  ─────────────
probs_max  = y_prob.max(axis=1)
thr = np.linspace(0,1,101)
prec, rec, f1 = [], [], []

for t in thr:
    m = probs_max >= t
    if m.sum()==0:
        p=r=1.0; f=0.0
    else:
        rep = classification_report(y_true[m], y_pred[m],
                                    output_dict=True, zero_division=0)
        p = rep["weighted avg"]["precision"]
        r = rep["weighted avg"]["recall"]
        f = rep["weighted avg"]["f1-score"]
    prec.append(p); rec.append(r); f1.append(f)

save_curve(thr, f1,  "Confidence", "Weighted F1",
           "F1 vs Confidence",          "F04_f1_confidence.png")
save_curve(thr, prec,"Confidence", "Weighted Precision",
           "Precision vs Confidence",   "F05_precision_confidence.png")
save_curve(rec, prec,"Recall",     "Precision",
           "Precision-Recall curve (macro-avg)",
           "F06_precision_recall.png")

print(f"📁  All figures saved to {OUT}")
