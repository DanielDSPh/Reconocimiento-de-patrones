import matplotlib.pyplot as plt
from numpy.random import random
from sklearn.model_selection import train_test_split
import pandas as pd
from kNNClass import kNN;

path_glass = "glass.csv"
path_diabetes = "diabetes.csv"
path_autoprice = "autoprice.csv"

data = pd.read_csv(path_autoprice)
data.head()
Y = data['class']
X = data.drop(['class'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.34, random_state=10)

knn = kNN(k=5, exp=2)
knn.fit(X_train, Y_train)

# Obtener predicciones
pred = knn.getDiscreteClassification(X_test)
print("Predicciones:", pred[:5])

# Obtener probabilidades de clase
probs = knn.getClassProbs(X_test)
print("\nProbabilidades:")
print(probs.head())