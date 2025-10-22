import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from kNNClassD import kNN  # tu clase con getPrediction

# --- Rutas ---
path_autoprice = "autoprice.csv"

# === 1) Cargar datos ===
df = pd.read_csv(path_autoprice)

# Toma 'price' como atributo de salida; si tu archivo usa otro nombre, cámbialo aquí.
target_col = 'class' if 'class' in df.columns else df.columns[-1]

Y = df[target_col]
X = df.drop(columns=[target_col])

# Opcional: si hay columnas no numéricas, conviértelas o elimínalas.
# Aquí descartamos no numéricas para mantener el ejemplo simple:
X = X.select_dtypes(include=[np.number]).copy()

# === 2) Hold-out ===
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.34, random_state=10
)

# === 3) Normalización (min-max) ===
# Usa stats de X_train para normalizar train y test
min_train = X_train.min()
range_train = (X_train.max() - min_train)
range_train[range_train == 0] = 1  # evitar división por cero

X_train_norm = (X_train - min_train) / range_train
X_test_norm  = (X_test  - min_train) / range_train

# === 4) Barrido de k y evaluación con MAE ===
k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

trainMAE = np.zeros(len(k_range))
testMAE  = np.zeros(len(k_range))

for idx, k in enumerate(k_range):
    reg = kNN(k=k, exp=2)
    reg.fit(X_train_norm, Y_train)

    # Predicciones con tu método de regresión (media de Y de los k vecinos)
    y_pred_train_df = reg.getPrediction(X_train_norm)
    y_pred_test_df  = reg.getPrediction(X_test_norm)

    # Si getPrediction devuelve DataFrame, toma la primera/única columna
    y_pred_train = y_pred_train_df.iloc[:, 0].to_numpy() if isinstance(y_pred_train_df, pd.DataFrame) else np.asarray(y_pred_train_df)
    y_pred_test  = y_pred_test_df.iloc[:, 0].to_numpy()  if isinstance(y_pred_test_df,  pd.DataFrame)  else np.asarray(y_pred_test_df)

    trainMAE[idx] = mean_absolute_error(Y_train, y_pred_train)
    testMAE[idx]  = mean_absolute_error(Y_test,  y_pred_test)

# === 5) Reporte de mejor k (menor MAE) ===
best_k_idx = np.argmin(testMAE)
best_k = k_range[best_k_idx]
print(f"Mejor k según MAE de Test: k={best_k}  (MAE_test={testMAE[best_k_idx]:.4f}, MAE_train={trainMAE[best_k_idx]:.4f})")

# === 6) Gráfica MAE vs k (más bajo es mejor) ===
plt.figure(figsize=(8,5))
plt.plot(k_range, trainMAE, 'o-', label='Train MAE')
plt.plot(k_range, testMAE,  's--', label='Test MAE')
plt.xlabel('k')
plt.ylabel('MAE')
plt.title('k-NN Regresión en AUTOPRICE (Métrica: MAE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
