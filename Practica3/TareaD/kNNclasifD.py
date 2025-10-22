import matplotlib.pyplot as plt
from numpy.random import random
from sklearn.model_selection import train_test_split
import pandas as pd
from kNNClassD import kNN;

data = pd.read_csv('autoprice.csv')
data.head()
Y = data['class']
X = data.drop(['class'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.34, random_state=10)


knn = kNN(k=5, exp=2)
knn.fit(X_train, Y_train)

y_pred_df = knn.getPrediction(X_test)
print(y_pred_df.head())