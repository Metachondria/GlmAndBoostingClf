from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Создание датасета для бинарной классификации
X, y = make_classification(n_samples=1000,  # Количество образцов
                           n_features=20,  # Количество признаков
                           n_classes=2,  # Два класса
                           n_informative=2,  # Количество информативных признаков
                           n_redundant=2,  # Количество избыточных признаков
                           random_state=42)  # Фиксируем random_state для воспроизводимости

X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
y = pd.Series(y, name='target')

scaler = StandardScaler()
X = scaler.fit_transform(X)


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
                else:
                    gradient = 2 * self.coef_regularization * self.weights_linear_combination

# Предсказания и оценка точности
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
predictionsSK = model2.predict(X_test)
accuracySK = accuracy_score(y_test, predictionsSK)

print(f'Accuracy: {accuracy}, \n {'-'* 400} \n AccuracySK: {accuracySK}')
