import matplotlib.pyplot as plt
import pandas as pd

# Class of k-Nearest Neigbor Classifier
class kNN():
    def __init__(self, k = 3, exp = 2):
    # constructor for kNN classifier
    # k is the number of neighbor for local class estimation
    # exp is the exponent for the Minkowski distance
        self.k = k
        self.exp = exp

    def fit(self, X_train, Y_train):
    # training k-NN method
    # X_train is the training data given with input attributes. n-th row correponds to n-th instance.
    # Y_train is the output data (output vector): n-th element of Y_train is the output value for n-th instance in X_train.
        self.X_train = X_train
        self.Y_train = Y_train

    def getDiscreteClassification(self, X_test):
    # predict-class k-NN method
    # X_test is the test data given with input attributes. Rows correpond to instances
    # Method outputs prediction vector Y_pred_test:  n-th element of Y_pred_test is the prediction for n-th instance in X_test

        Y_pred_test = [] #prediction vector Y_pred_test for all the test instances in X_test is initialized to empty list []


        for i in range(len(X_test)):   #iterate over all instances in X_test
            test_instance = X_test.iloc[i] #i-th test instance

            distances = []  #list of distances of the i-th test_instance for all the train_instance s in X_train, initially empty.

            for j in range(len(self.X_train)):  #iterate over all instances in X_train
                train_instance = self.X_train.iloc[j] #j-th training instance
                distance = self.Minkowski_distance(test_instance, train_instance) #distance between i-th test instance and j-th training instance
                distances.append(distance) #add the distance to the list of distances of the i-th test_instance

            # Store distances in a dataframe. The dataframe has the index of Y_train in order to keep the correspondence with the classes of the training instances
            df_dists = pd.DataFrame(data=distances, columns=['dist'], index = self.Y_train.index)

            # Sort distances, and only consider the k closest points in the new dataframe df_knn
            df_nn = df_dists.sort_values(by=['dist'], axis=0)
            df_knn =  df_nn[:self.k]

            # Note that the index df_knn.index of df_knn contains indices in Y_train of the k-closed training instances to
            # the i-th test instance. Thus, the dataframe self.Y_train[df_knn.index] contains the classes of those k-closed
            # training instances. Method value_counts() computes the counts (number of occurencies) for each class in
            # self.Y_train[df_knn.index] in dataframe predictions.
            predictions = self.Y_train[df_knn.index].value_counts()

            # the first element of the index predictions.index contains the class with the highest count; i.e. the prediction y_pred_test.
            y_pred_test = predictions.index[0]

            # add the prediction y_pred_test to the prediction vector Y_pred_test for all the test instances in X_test
            Y_pred_test.append(y_pred_test)

        return Y_pred_test


    def Minkowski_distance(self, x1, x2):
        # computes the Minkowski distance of x1 and x2 for two labeled instances (x1,y1) and (x2,y2)
        distance = (abs(x1 - x2)**self.exp).sum()
        distance = distance**(1/self.exp)

        return distance


    def normalize(self, data):
        normalized_data = (data-data.min())/(data.max()-data.min())
        return normalized_data
    
    def getPrediction(self, X_test):
        """
        k-NN regresión: para cada instancia en X_test devuelve
        la media de Y_train de sus k vecinos más cercanos.
        Retorna un DataFrame con el mismo índice que X_test.
        """
        preds = []

        for i in range(len(X_test)):
            test_instance = X_test.iloc[i]

            # Distancias a todo el set de entrenamiento (misma métrica que usas en clasificación)
            distances = []
            for j in range(len(self.X_train)):
                train_instance = self.X_train.iloc[j]
                d = self.Minkowski_distance(test_instance, train_instance)
                distances.append(d)

            # Tomamos los k vecinos más cercanos
            df_dists = pd.DataFrame({"dist": distances}, index=self.Y_train.index)
            df_knn = df_dists.nsmallest(self.k, "dist")

            # Valores de salida de esos vecinos
            y_neighbors = self.Y_train.loc[df_knn.index]

            # Media (si Y es Serie -> escalar; si Y es DataFrame -> vector por columnas)
            if isinstance(y_neighbors, pd.Series):
                pred = y_neighbors.mean()
            else:  # múltiples columnas en Y_train
                pred = y_neighbors.mean(axis=0)

            preds.append(pred)

        # Convertimos a DataFrame y alineamos índice con X_test
        if isinstance(self.Y_train, pd.Series):
            col_name = self.Y_train.name or "prediction"
            preds_df = pd.DataFrame(preds, columns=[col_name], index=X_test.index)
        else:
            preds_df = pd.DataFrame(preds, columns=self.Y_train.columns, index=X_test.index)

        return preds_df
