# export_results.py  ——  运行前确保已 pip install pandas
import numpy as np, pandas as pd, datetime, zipfile, pathlib, shutil, os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

ROOT   = r'C:\Users\Cleveland\Desktop\mechanical\assignment5'
OUT    = pathlib.Path(ROOT, 'result')
BEST_W = OUT / 'best_stage2.weights.h5'   # ← 若脚本选了 stage1，请改名

# ───────────────── 1  重新构建数据集 & 模型 ─────────────────
IMG_SIZE = (224,224); BATCH=32
PREP = tf.keras.applications.efficientnet.preprocess_input

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
             preprocessing_function=PREP)
test_ds  = test_gen.flow_from_directory(
             pathlib.Path(ROOT,'test'), target_size=IMG_SIZE,
             batch_size=BATCH, shuffle=False)

CLASSES = list(test_ds.class_indices.keys())

base = tf.keras.applications.EfficientNetB0(
         include_top=False, input_shape=IMG_SIZE+(3,), weights=None)
model = tf.keras.Sequential([
    base, tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])
model.load_weights(BEST_W)

# ───────────────── 2  预测 & 指标表 ─────────────────
y_true = test_ds.classes
y_prob = model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)

cm   = confusion_matrix(y_true, y_pred)
rep  = classification_report(
         y_true, y_pred, target_names=CLASSES, output_dict=True)

# DataFrame（每类 + macro + weighted + accuracy）
rows  = CLASSES + ['macro avg','weighted avg']
data  = {k:[rep[k]['precision'], rep[k]['recall'], rep[k]['f1-score']]
         for k in rows if k in rep}
df    = pd.DataFrame(data, index=['precision','recall','f1']).T
df['support'] = [rep[k]['support'] for k in rows]

acc = np.mean(y_true == y_pred)
df.loc['accuracy (%)','f1'] = acc * 100

csv_path = OUT/'metrics_summary.csv'
df.to_csv(csv_path, float_format='%.4f')
np.save(OUT/'confusion_matrix.npy', cm)

print(f'✅  CSV saved to  {csv_path}')

# ───────────────── 3  打包 ZIP ─────────────────
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
zip_path  = OUT/f'results_{timestamp}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    # a) 图表 PNG
    for png in OUT.glob('F0*.png'):
        z.write(png, arcname=png.name)
    # b) 指标 CSV + NPY
    z.write(csv_path, arcname=csv_path.name)
    z.write(OUT/'confusion_matrix.npy', arcname='confusion_matrix.npy')
    # c) 错误图片文件夹（如果有）
    err_dir = OUT/'errors'
    if err_dir.exists():
        for p in err_dir.glob('*'):
            z.write(p, arcname=f'errors/{p.name}')

print(f'📦  All results zipped → {zip_path}')
print('   现在可直接在资源管理器中右键下载或复制。')
