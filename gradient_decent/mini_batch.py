import random
from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


X, y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# fitting the sklearn model
reg = LinearRegression()
reg.fit(X_train, y_train)
y_reg_pred = reg.predict(X_train)

print("=" * 50)
print(reg.coef_)
print(reg.intercept_)
print("=" * 50)


class MBGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=100):
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        # init coefs and intercept_
        self.intercept_ = 0.2
        self.coef_ = np.ones(X.shape[1])

        for i in range(self.epochs):
            for j in range(int(X_train.shape[0] / self.batch_size)):
                idx = random.sample(range(X_train.shape[0]), self.batch_size)

                # predicting y hat
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_

                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.learning_rate * intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef_ = self.coef_ - (self.learning_rate * coef_der)
        print("Coef: ", self.coef_)
        print("Intercept: ", self.intercept_)

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_


mini_batch = MBGDRegressor(
    batch_size=int(X_train.shape[0] / 5), learning_rate=0.01, epochs=100
)
mini_batch.fit(X_train, y_train)
y_pred = mini_batch.predict(X_test)


# ==========================
print("R2_Score: ", r2_score(y_test, y_pred))
