# Hello ML Models

This folder contains simple models trained from scratch using Python:

- Logic gate classifiers with a sigmoid neuron:
  - `and_gate.py`
  - `or_gates.py`
  - `nand_gate.py`
- A linear regression model:
  - `main.py`

## Requirements

- Python 3
- Standard library only (`math`, `random`)

## How to Run

From this directory:

```bash
python and_gate.py
python or_gates.py
python nand_gate.py
python main.py
```

## Training Approach

All models are trained using numerical gradients (finite differences):

- Gradient estimate: `(f(param + eps) - f(param)) / eps`
- Update rule: `param = param - learning_rate * gradient`
- Loss: mean squared error (MSE)

## Models

### 1) AND Gate (`and_gate.py`)

Truth table used for training:

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

Details:

- Model: `sigmoid(x1*w1 + x2*w2 + b)`
- Epochs: `10000`
- Learning rate: `0.1`
- `eps`: `1e-3`
- Final output includes:
  - probability prediction
  - class prediction using threshold `0.5`
  - full dataset accuracy

Observed run output:

- Accuracy: `100.00% (4/4)`

### 2) OR Gate (`or_gates.py`)

Truth table used for training:

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 1 |

Details:

- Model: `sigmoid(x1*w1 + x2*w2 + b)`
- Epochs: `10000`
- Learning rate: `0.1`
- `eps`: `1e-3`
- Final output includes probability, class, and full dataset accuracy

Observed run output:

- Accuracy: `100.00% (4/4)`

### 3) NAND Gate (`nand_gate.py`)

Truth table used for training:

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 1 | 0 |

Details:

- Model: `sigmoid(x1*w1 + x2*w2 + b)`
- Epochs: `10000`
- Learning rate: `0.1`
- `eps`: `1e-3`
- Final output includes probability, class, and full dataset accuracy

Observed run output:

- Accuracy: `100.00% (4/4)`

### 4) Linear Regression (`main.py`)

Training data:

| x | y |
|---|---|
| 0 | 0 |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |

Target relation is approximately:

- `y = 2x`

Details:

- Model: `y_hat = x*w + b`
- Epochs: `4000`
- Learning rate: `1e-3`
- `eps`: `1e-3`
- Final output prints learned `weightage` and `bias`

Observed run output:

- `weightage` near `2.0`
- `bias` near `0.0`

## Notes

- Because parameters are initialized randomly, exact final values can vary between runs.
- Gate models are evaluated on the full training truth table and print full-dataset accuracy.
