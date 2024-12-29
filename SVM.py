from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=2, n_redundant=2, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
y = pd.Series(y, name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class SVM:
    def __init__(self, learning_rate=0.001, n_iter=1000, coef_regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights_linear_combination = None
        self.coef_regularization = coef_regularization

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)
        self.weights_linear_combination = np.zeros(n_features)
        for _ in range(self.n_iter):
            for indx, x_i in enumerate(X):
                linear_comb = np.dot(x_i, self.weights_linear_combination)
                margin = y[indx] * linear_comb
                if margin < 1:
                    gradient = -y[indx] * x_i + 2 * self.coef_regularization * self.weights_linear_combination
                    self.weights_linear_combination -= self.learning_rate * gradient
                else:
                    gradient = 2 * self.coef_regularization * self.weights_linear_combination
                    self.weights_linear_combination -= self.learning_rate * gradient

    def predict(self, X):
        linear_comb = np.dot(X, self.weights_linear_combination)
        return np.sign(linear_comb)

model = SVM(learning_rate=0.001, n_iter=1000, coef_regularization=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

from sklearn.svm import SVC
model2 = SVC(kernel='linear')
model2.fit(X_train, y_train)
predictionsSK = model2.predict(X_test)
accuracySK = accuracy_score(y_test, predictionsSK)

print(f'Accuracy: {accuracy}, \n {"-"* 400} \n AccuracySK: {accuracySK}')
