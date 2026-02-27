import math
import random


# XOR Gate
train = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

train_count = len(train)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict(
    w_ih: list[list[float]],
    b_h: list[float],
    w_ho: list[float],
    b_o: float,
    x1: float,
    x2: float,
) -> float:
    h1 = sigmoid(x1 * w_ih[0][0] + x2 * w_ih[0][1] + b_h[0])
    h2 = sigmoid(x1 * w_ih[1][0] + x2 * w_ih[1][1] + b_h[1])
    return sigmoid(h1 * w_ho[0] + h2 * w_ho[1] + b_o)


def loss(
    w_ih: list[list[float]],
    b_h: list[float],
    w_ho: list[float],
    b_o: float,
) -> float:
    result = 0.0
    for x1, x2, y in train:
        y_hat = predict(w_ih, b_h, w_ho, b_o, x1, x2)
        d = y - y_hat
        result += d * d
    return result / train_count


def accuracy(
    w_ih: list[list[float]],
    b_h: list[float],
    w_ho: list[float],
    b_o: float,
) -> float:
    correct = 0
    for x1, x2, y in train:
        predicted = 1 if predict(w_ih, b_h, w_ho, b_o, x1, x2) >= 0.5 else 0
        if predicted == int(y):
            correct += 1
    return correct / train_count


if __name__ == "__main__":
    random.seed(42)

    # 2 -> 2 -> 1 network parameters
    w_ih = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] for _ in range(2)]
    b_h = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    w_ho = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
    b_o = random.uniform(-1.0, 1.0)

    learning_rate = 1.0
    epochs = 20000

    for epoch in range(epochs):
        grad_w_ih = [[0.0, 0.0], [0.0, 0.0]]
        grad_b_h = [0.0, 0.0]
        grad_w_ho = [0.0, 0.0]
        grad_b_o = 0.0

        for x1, x2, y in train:
            # Forward pass
            h1_in = x1 * w_ih[0][0] + x2 * w_ih[0][1] + b_h[0]
            h2_in = x1 * w_ih[1][0] + x2 * w_ih[1][1] + b_h[1]
            h1 = sigmoid(h1_in)
            h2 = sigmoid(h2_in)

            o_in = h1 * w_ho[0] + h2 * w_ho[1] + b_o
            y_hat = sigmoid(o_in)

            # Backward pass (MSE + sigmoid)
            d_loss_y_hat = 2.0 * (y_hat - y)
            delta_o = d_loss_y_hat * y_hat * (1.0 - y_hat)

            grad_w_ho[0] += delta_o * h1
            grad_w_ho[1] += delta_o * h2
            grad_b_o += delta_o

            delta_h1 = delta_o * w_ho[0] * h1 * (1.0 - h1)
            delta_h2 = delta_o * w_ho[1] * h2 * (1.0 - h2)

            grad_w_ih[0][0] += delta_h1 * x1
            grad_w_ih[0][1] += delta_h1 * x2
            grad_b_h[0] += delta_h1

            grad_w_ih[1][0] += delta_h2 * x1
            grad_w_ih[1][1] += delta_h2 * x2
            grad_b_h[1] += delta_h2

        # Mean gradient update
        scale = 1.0 / train_count
        for i in range(2):
            for j in range(2):
                w_ih[i][j] -= learning_rate * grad_w_ih[i][j] * scale
            b_h[i] -= learning_rate * grad_b_h[i] * scale
        for i in range(2):
            w_ho[i] -= learning_rate * grad_w_ho[i] * scale
        b_o -= learning_rate * grad_b_o * scale

        if epoch % 2000 == 0:
            print(f"epoch: {epoch}, loss: {loss(w_ih, b_h, w_ho, b_o):.6f}")

    print("\n--- Predictions ---")
    for x1, x2, expected in train:
        y_hat = predict(w_ih, b_h, w_ho, b_o, x1, x2)
        predicted_class = 1 if y_hat >= 0.5 else 0
        print(
            f"Input: [{int(x1)}, {int(x2)}], Expected: {int(expected)}, "
            f"Predicted: {y_hat:.4f}, Class: {predicted_class}"
        )

    model_accuracy = accuracy(w_ih, b_h, w_ho, b_o)
    correct_count = int(model_accuracy * train_count)
    print(
        f"\nModel accuracy on full dataset: "
        f"{model_accuracy * 100:.2f}% ({correct_count}/{train_count})"
    )
