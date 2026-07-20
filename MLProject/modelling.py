"""
modelling.py (versi untuk MLflow Project / CI)

Perbedaan dengan modelling.py di Kriteria 2:
- Tidak diarahkan ke server MLflow manapun (tanpa set_tracking_uri),
  jadi otomatis mencatat ke folder lokal ./mlruns di dalam runner CI.
  Ini karena di GitHub Actions tidak ada server MLflow yang menyala terus-menerus.
"""

import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.autolog()

train_df = pd.read_csv("namadataset_preprocessing/train.csv")
test_df = pd.read_csv("namadataset_preprocessing/test.csv")

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

print("Data latih :", X_train.shape)
print("Data uji   :", X_test.shape)

# CATATAN: Tidak perlu "with mlflow.start_run():" di sini.
# Saat dijalankan lewat perintah "mlflow run .", MLflow SUDAH otomatis
# membuka sebuah run untuk kita. Kalau kita buka run lagi secara manual
# di dalam skrip ini, akan tabrakan (error run ID mismatch).
# Jadi cukup langsung latih modelnya, autolog akan otomatis mencatat
# ke run yang sudah dibuka oleh "mlflow run" tadi.

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("Training selesai. Hasil tersimpan di folder ./mlruns")
