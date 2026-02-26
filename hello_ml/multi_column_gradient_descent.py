import random


# Dataset format:
# [hours_studied, sleep_hours, practice_tests, exam_score]
DATA = [
    [2, 7, 1, 55],
    [3, 6, 1, 58],
    [4, 7, 2, 64],
    [5, 6, 2, 67],
    [6, 8, 3, 74],
    [7, 7, 3, 78],
    [8, 8, 4, 84],
    [9, 7, 4, 88],
    [10, 8, 5, 93],
]

FEATURE_NAMES = ["hours_studied", "sleep_hours", "practice_tests"]
TARGET_NAME = "exam_score"


def split_features_and_target(data: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """Split table rows into X (input columns) and y (prediction column)."""
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    for row in data:
        x_rows.append(row[:-1])
        y_values.append(row[-1])
    return x_rows, y_values


def standardize_matrix(x_rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """
    Standardize each input column to make gradient descent stable:
    scaled = (value - mean) / std
    """
    row_count = len(x_rows)
    col_count = len(x_rows[0])

    means: list[float] = [0.0] * col_count
    stds: list[float] = [0.0] * col_count

    # Compute column means
    for col in range(col_count):
        means[col] = sum(row[col] for row in x_rows) / row_count

    # Compute column standard deviations
    for col in range(col_count):
        variance = sum((row[col] - means[col]) ** 2 for row in x_rows) / row_count
        stds[col] = variance ** 0.5
        if stds[col] == 0:
            stds[col] = 1.0

    # Build standardized matrix
    x_scaled: list[list[float]] = []
    for row in x_rows:
        x_scaled.append([(row[col] - means[col]) / stds[col] for col in range(col_count)])

    return x_scaled, means, stds


def standardize_vector(y_values: list[float]) -> tuple[list[float], float, float]:
    """Standardize target column for smoother training updates."""
    count = len(y_values)
    mean_y = sum(y_values) / count
    variance_y = sum((y - mean_y) ** 2 for y in y_values) / count
    std_y = variance_y ** 0.5
    if std_y == 0:
        std_y = 1.0

    y_scaled = [(y - mean_y) / std_y for y in y_values]
    return y_scaled, mean_y, std_y


def predict(weights: list[float], bias: float, x_row: list[float]) -> float:
    """Linear model prediction: y = w1*x1 + w2*x2 + ... + b"""
    return sum(w * x for w, x in zip(weights, x_row)) + bias


def cost_func(x_rows: list[list[float]], y_values: list[float], weights: list[float], bias: float) -> float:
    """Mean squared error over the full dataset."""
    m = len(x_rows)
    total_error = 0.0
    for x_row, y_true in zip(x_rows, y_values):
        y_pred = predict(weights, bias, x_row)
        total_error += (y_pred - y_true) ** 2
    return total_error / m


def gradients(
    x_rows: list[list[float]],
    y_values: list[float],
    weights: list[float],
    bias: float,
) -> tuple[list[float], float]:
    """Analytical gradients for MSE in multi-column linear regression."""
    m = len(x_rows)
    n = len(weights)
    dw: list[float] = [0.0] * n
    db = 0.0

    for x_row, y_true in zip(x_rows, y_values):
        y_pred = predict(weights, bias, x_row)
        error = y_pred - y_true

        for j in range(n):
            dw[j] += (2.0 / m) * error * x_row[j]
        db += (2.0 / m) * error

    return dw, db


def train_model(
    x_rows: list[list[float]],
    y_values: list[float],
    epochs: int = 4000,
    learning_rate: float = 0.05,
    log_step: int = 500,
) -> tuple[list[float], float]:
    """Train weights and bias using gradient descent."""
    feature_count = len(x_rows[0])
    weights = [random.uniform(-0.5, 0.5) for _ in range(feature_count)]
    bias = random.uniform(-0.5, 0.5)

    for step in range(epochs):
        current_cost = cost_func(x_rows, y_values, weights, bias)
        if step % log_step == 0 or step == epochs - 1:
            print(f"step={step:4d} cost={current_cost:.6f} weights={weights} bias={bias:.6f}")

        dw, db = gradients(x_rows, y_values, weights, bias)
        weights = [w - learning_rate * grad_w for w, grad_w in zip(weights, dw)]
        bias -= learning_rate * db

    return weights, bias


def scale_new_row(x_row: list[float], means: list[float], stds: list[float]) -> list[float]:
    """Use training statistics to scale any new input row."""
    return [(x_row[i] - means[i]) / stds[i] for i in range(len(x_row))]


if __name__ == "__main__":
    random.seed(69)

    # 1) Convert full table into X and y
    x_raw, y_raw = split_features_and_target(DATA)

    # 2) Scale both input columns and target column
    x_scaled, x_means, x_stds = standardize_matrix(x_raw)
    y_scaled, y_mean, y_std = standardize_vector(y_raw)

    # 3) Train model by gradient descent
    final_weights, final_bias = train_model(x_scaled, y_scaled)

    # 4) Show predictions on training rows (converted back to original score scale)
    print("\n--- Training Predictions ---")
    for x_row_raw, y_true in zip(x_raw, y_raw):
        x_row_scaled = scale_new_row(x_row_raw, x_means, x_stds)
        y_pred_scaled = predict(final_weights, final_bias, x_row_scaled)
        y_pred_original = y_pred_scaled * y_std + y_mean
        print(
            f"Input={x_row_raw} | Expected {TARGET_NAME}={y_true:.1f} | "
            f"Predicted {TARGET_NAME}={y_pred_original:.2f}"
        )

    # 5) Predict for one new sample
    new_student = [7.5, 7.0, 3.0]  # [hours_studied, sleep_hours, practice_tests]
    new_student_scaled = scale_new_row(new_student, x_means, x_stds)
    new_score_scaled = predict(final_weights, final_bias, new_student_scaled)
    new_score = new_score_scaled * y_std + y_mean

    print(
        f"\nNew input {new_student} -> Predicted {TARGET_NAME}: {new_score:.2f}"
    )

    # Detailed self-explanatory document block (triple-quoted f-string)
    description = f"""
Detailed Description
--------------------
This file demonstrates gradient descent with multiple input columns
and one prediction column.

Input columns:
1) {FEATURE_NAMES[0]}
2) {FEATURE_NAMES[1]}
3) {FEATURE_NAMES[2]}

Prediction column:
1) {TARGET_NAME}

How it works:
1) We split the table into features (X) and target (y).
2) We standardize X and y, so updates are stable and training is fast.
3) We use a linear model:
      y_pred = w1*x1 + w2*x2 + w3*x3 + b
4) We calculate MSE cost and its gradients.
5) We update parameters using gradient descent:
      w = w - learning_rate * dw
      b = b - learning_rate * db

Final learned parameters (on standardized data):
- weights = {[round(w, 4) for w in final_weights]}
- bias    = {final_bias:.4f}

Example prediction:
- Input row: {new_student}
- Predicted {TARGET_NAME}: {new_score:.2f}

This example is simple by design, so you can add/remove columns or
change learning rate and epochs to observe training behavior.
"""
    print(description)
