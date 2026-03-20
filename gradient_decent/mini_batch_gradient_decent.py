import numpy as np
import pandas as pd

# 1. Setup Dummy Data (Following your naming convention)
data = {
    'iq': [110, 120, 105, 115, 125, 130, 110, 108],
    'gender': [1, 0, 1, 0, 1, 1, 0, 1], # Assuming 1 for Male, 0 for Female
    'age': [25, 32, 45, 22, 38, 29, 31, 40],
    'experience': [2, 10, 20, 1, 15, 5, 8, 18],
    'lap': [15, 25, 35, 12, 30, 20, 22, 32] # Target Variable
}

df = pd.DataFrame(data)

# 2. Parameters
learning_rate = 0.01
epochs = 50
batch_size = 2
n_samples = len(df)

# Initialize weights (4 features + 1 bias)
weights = np.zeros(4)
bias = 0

# 3. Mini-Batch Gradient Descent Loop
X = df[['iq', 'gender', 'age', 'experience']].values
y = df['lap'].values

for epoch in range(epochs):
    # Shuffle data at the start of each epoch
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for i in range(0, n_samples, batch_size):
        # Extract the mini-batch
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Calculate Predictions
        y_pred = np.dot(X_batch, weights) + bias
        
        # Calculate Gradients (Mean for the batch)
        error = y_pred - y_batch
        dw = (1/batch_size) * np.dot(X_batch.T, error)
        db = (1/batch_size) * np.sum(error)
        
        # Update Weights
        weights -= learning_rate * dw
        bias -= learning_rate * db

print(f"Optimized Weights: {weights}")
print(f"Optimized Bias: {bias}")