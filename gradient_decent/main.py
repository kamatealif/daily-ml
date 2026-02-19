from sklearn.datasets import make_regression

import matplotlib.pyplot as plt 
import numpy as np 


X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20)


plt.scatter(X,y)

class GDregressor:
    def __init__(self, learning_rate=0.001, epochs=1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0.0
        self.b = 0.0
        self.loss_history = []
        
    def fit(self, X, y):
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)
        n = X.shape[0]

        for i in range(self.epochs):
            y_pred = self.m * X + self.b
            error = y_pred - y

            m_slope = (2 / n) * np.dot(error, X)
            b_slope = (2 / n) * np.sum(error)

            self.m = self.m - (self.learning_rate * m_slope)
            self.b = self.b - (self.learning_rate * b_slope)

            self.loss_history.append(np.mean(error ** 2))

        return self

    def predict(self, X):
        X = np.asarray(X).reshape(-1)
        return self.m * X + self.b

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
gd = GDregressor(0.001, 10000)
gd.fit(X,y)
print(f"m={gd.m:.3f}, b={gd.b:.3f}, r2={gd.score(X, y):.3f}")
