import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from kNN_Class import kNN; 

path_glass = "glass.csv"
path_diabetes = "diabetes.csv"
path_autoprice = "autoprice.csv"

df = pd.read_csv(path_glass)
df.head()

##################################################
# Hold-out testing: Training and Test set creation
##################################################

data = pd.read_csv(path_glass)
data.head()
Y = data['class']
X = data.drop(['class'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.34, random_state=10)


# range for the values of parameter k for kNN

k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

# --- 1. CASO SIN NORMALIZAR ---
# (Usando tu primer bucle comentado, que está correcto)

print("Ejecutando SIN normalización...")
trainAcc_unnorm = np.zeros(len(k_range))
testAcc_unnorm = np.zeros(len(k_range))
index = 0

for k in k_range:
    clf = kNN(k)
    clf.fit(X_train, Y_train) # <- Datos originales
    Y_predTrain = clf.getDiscreteClassification(X_train) # <- Datos originales
    Y_predTest = clf.getDiscreteClassification(X_test) # <- Datos originales

    trainAcc_unnorm[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc_unnorm[index] = accuracy_score(Y_test, Y_predTest)
    index += 1

# --- 2. CASO CON NORMALIZACIÓN (La Forma Correcta) ---
# Aquí aplicamos la normalización ANTES del bucle

print("Ejecutando CON normalización...")

# 2a. Normaliza los datos CORRECTAMENTE
# Usamos una instancia temporal solo para acceder al método normalize
temp_clf = kNN()
X_train_norm = temp_clf.normalize(X_train) # Normaliza X_train usando sus propios stats

# 2b. ¡¡AQUÍ ESTÁ EL PASO CLAVE!!
# Normaliza X_test usando los stats de X_train
min_train = X_train.min()
max_train = X_train.max()
range_train = max_train - min_train
range_train[range_train == 0] = 1 # Evitar división por cero

X_test_norm = (X_test - min_train) / range_train

# 2c. Ahora ejecuta el bucle con los datos YA NORMALIZADOS
trainAcc_norm = np.zeros(len(k_range))
testAcc_norm = np.zeros(len(k_range))
index = 0

for k in k_range:
    clf = kNN(k)
    # 2d. Usa los datos normalizados en fit y getDiscreteClassification
    clf.fit(X_train_norm, Y_train)
    Y_predTrain = clf.getDiscreteClassification(X_train_norm)
    Y_predTest = clf.getDiscreteClassification(X_test_norm)

    trainAcc_norm[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc_norm[index] = accuracy_score(Y_test, Y_predTest)
    index += 1


# --- 3. GRAFICAR AMBOS RESULTADOS ---

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, trainAcc_unnorm, 'ro-', label='Train (sin norm)')
plt.plot(k_range, testAcc_unnorm, 'bv--', label='Test (sin norm)')
plt.title('Sin Normalización')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_range, trainAcc_norm, 'ro-', label='Train (con norm)')
plt.plot(k_range, testAcc_norm, 'bv--', label='Test (con norm)')
plt.title('Con Normalización')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

