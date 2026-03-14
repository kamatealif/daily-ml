"""
Stochastic Gradient Descent (SGD) Example

Unlike Batch Gradient Descent which uses all samples to compute gradients each epoch,
SGD updates weights after each individual sample. This makes it:
- Much faster per epoch (no full pass over data per update)
- Noisier updates (can help escape shallow local minima)
- Better suited for large datasets and online learning
"""

import numpy as np


class StandardScaler:
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StochasticGradientDescentRegressor:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StochasticGradientDescentRegressor":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        indices = np.arange(n_samples)

        for epoch in range(self.epochs):
            # Shuffle data each epoch for stochasticity
            np.random.shuffle(indices)
            epoch_loss = 0.0

            for i in indices:
                xi = X[i : i + 1]  # shape (1, n_features)
                yi = y[i : i + 1]  # shape (1,)

                # Predict for single sample
                y_pred = xi @ self.weights + self.bias
                error = y_pred - yi

                # SGD: gradient from ONE sample only
                dw = 2.0 * (xi.T @ error)
                db = 2.0 * np.sum(error)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                epoch_loss += np.mean(error**2)

            self.loss_history.append(epoch_loss / n_samples)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


def encode_gender(gender: str) -> int:
    return 1 if gender.strip().lower() == "male" else 0


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    # [cgpa, iq, gender], salary_lpa
    rows = [
        (9.1, 142, "male", 23.8),
        (8.7, 136, "female", 21.4),
        (8.2, 130, "male", 20.8),
        (7.9, 125, "female", 18.7),
        (9.4, 145, "male", 24.6),
        (7.1, 110, "female", 15.0),
        (8.0, 128, "male", 19.2),
        (6.8, 105, "female", 14.2),
        (9.0, 138, "female", 22.1),
        (7.6, 118, "male", 17.8),
        (8.5, 134, "female", 20.9),
        (6.9, 108, "male", 15.3),
        (9.3, 146, "female", 24.0),
        (7.3, 114, "male", 16.8),
        (8.8, 139, "male", 22.7),
        (7.4, 116, "female", 16.4),
    ]

    X = np.array([[cgpa, iq, encode_gender(gender)] for cgpa, iq, gender, _ in rows], dtype=float)
    y = np.array([salary for *_, salary in rows], dtype=float)
    return X, y


def main() -> None:
    np.random.seed(42)

    X_raw, y = build_dataset()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # SGD typically needs more epochs (one update per sample) but smaller learning rate
    model = StochasticGradientDescentRegressor(learning_rate=0.01, epochs=500)
    model.fit(X, y)

    y_pred = model.predict(X)
    train_r2 = model.r2_score(y, y_pred)

    print("Stochastic Gradient Descent Salary Model")
    print(f"weights (scaled features) = {model.weights}")
    print(f"bias = {model.bias:.4f}")
    print(f"final MSE = {model.loss_history[-1]:.4f}")
    print(f"train R2 = {train_r2:.4f}")

    new_employee = np.array([[8.3, 132, encode_gender("female")]], dtype=float)
    new_employee_scaled = scaler.transform(new_employee)
    predicted_salary = model.predict(new_employee_scaled)[0]
    print(f"Predicted salary for CGPA=8.3, IQ=132, Gender=female: {predicted_salary:.2f} LPA")


if __name__ == "__main__":
    main()
