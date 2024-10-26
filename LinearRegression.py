import numpy as np
import pandas as pd
import sklearn.linear_model

from sklearn.metrics import root_mean_squared_error

class LinearRegression:
    def __init__(self,learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_linear_combinations = None

    def fit(self,X,y):
        m = X.shape[0]
        n = X.shape[1]
        vars_linear_combination = np.c_[np.ones(m), X.values]
        self.weights_linear_combinations = np.zeros(n + 1)
        for _ in range(self.n_iterations):
            y_pred = np.dot(vars_linear_combination, self.weights_linear_combinations.T)
            error = y - y_pred
            gradients = -(2/m) * np.dot(vars_linear_combination.T,error)
            self.weights_linear_combinations -= gradients * self.learning_rate
    def predict(self,X):
        m = X.shape[0]
        vars_linear_combination = np.c_[np.ones(m), X.values]
        return np.dot(vars_linear_combination, self.weights_linear_combinations.T)


my_model = LinearRegression()
sk_model = sklearn.linear_model.LinearRegression()

df = pd.read_csv('data2.csv')
X = df.drop(columns='Target')
y = df.Target

my_model.fit(X,y)
sk_model.fit(X,y)

print(f'Score my model:{root_mean_squared_error(y,my_model.predict(X))}')
print(f'Score sklearn model:{root_mean_squared_error(y,sk_model.predict(X))}')