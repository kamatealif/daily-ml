import random 
import math 


# AND Gate
train = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1]
]

train_count = len(train)

def sigmoid(x:float):
    return 1 / (1 + math.exp(-x))

def cost_func(w1, w2, b):
    result = 0
    for row in train:
        x1: float = row[0]
        x2: float = row[1]
        y: float = row[2]
        y_hat = sigmoid(x1*w1 + x2*w2 + b)
        d = y - y_hat
        result += d * d 
        
    result /= train_count
    return result 

def random_num (end: float = 1.0):
    return random.uniform(0, end)

def predict(w1: float, w2: float, b: float, x1: float, x2: float):
    return sigmoid(x1 * w1 + x2 * w2 + b)

def accuracy(w1: float, w2: float, b: float):
    correct = 0
    for row in train:
        x1: float = row[0]
        x2: float = row[1]
        expected: float = row[2]
        predicted: int = 1 if predict(w1, w2, b, x1, x2) >= 0.5 else 0
        if predicted == expected:
            correct += 1
    return correct / train_count


if __name__ == "__main__":
    # random.seed(69)
    w1:float = random_num()
    w2:float = random_num()
    b:float = random_num()
    
    eps: float = 1e-3
    rate: float = 1e-1
    
    for i in range(10000):
        cost = cost_func(w1, w2, b)
        print(f"w1 : {w1}, w2 : {w2}, b : {b}, cost = {cost}")
        
        dw1: float = (cost_func(w1 + eps, w2, b) - cost) / eps
        dw2: float = (cost_func(w1, w2 + eps, b) - cost) / eps
        db: float = (cost_func(w1, w2, b + eps) - cost) / eps
        
        w1 -= rate * dw1
        w2 -= rate * dw2
        b -= rate * db
        
         # Predict on training set
    print("\n--- Predictions ---")
    for row in train:
        x1, x2, expected = row[0], row[1], row[2]
        prediction = predict(w1, w2, b, x1, x2)
        predicted_class = 1 if prediction >= 0.5 else 0
        print(
            f"Input: [{x1}, {x2}], Expected: {expected}, "
            f"Predicted: {prediction:.4f}, Class: {predicted_class}"
        )

    model_accuracy = accuracy(w1, w2, b)
    correct_count = int(model_accuracy * train_count)
    print(
        f"\nModel accuracy on full dataset: "
        f"{model_accuracy * 100:.2f}% ({correct_count}/{train_count})"
    )
