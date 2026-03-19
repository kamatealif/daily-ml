# Gradient Descent Models

This repository is a small learning project around **linear regression** and the different ways we can train it.

Instead of using only one library model, the project shows the same core idea in multiple forms:

- a hand-written gradient descent regressor in `main.py`
- a full **Batch Gradient Descent** regressor for salary prediction
- a full **Stochastic Gradient Descent** regressor for salary prediction
- **scikit-learn `LinearRegression`** as a baseline inside the notebooks

The goal is to understand not just how to call a model, but **how the model actually learns**.

## What Is The Common Idea?

All of the custom models here are solving a **regression** problem.

Regression means:

- we want to predict a continuous value
- examples in this repo include values like salary or diabetes progression

The model tries to learn a line or hyperplane:

```text
y = w1x1 + w2x2 + ... + b
```

Where:

- `x` = input features
- `w` = learned weights
- `b` = bias / intercept
- `y` = predicted output

Training works by:

1. making a prediction
2. measuring the error
3. computing gradients
4. updating weights and bias to reduce the error

That update process is called **gradient descent**.

## Models In This Repo

### 1. `GDregressor` in `main.py`

**What it does**

This is the simplest custom model in the repo. It learns a straight-line relationship for a dataset with **one input feature**.

**How it works**

- starts with `m = 0` and `b = 0`
- predicts with `y = mX + b`
- computes error using mean squared error style gradients
- updates:
  - slope `m`
  - intercept `b`
- repeats for many epochs until the line fits the data better

This is standard gradient descent for a **single-variable linear regression** model.

**Why use it**

- best for learning the basics
- easy to read and understand
- shows clearly how slope and intercept are updated

**Why not use it for bigger projects**

- only handles one feature cleanly
- no train/test workflow
- no scaling step
- built for learning, not production

### 2. `BatchGradientDescentRegressor`

File: `batch_gradient_descent_salary_example.py`

**What it does**

This model predicts salary using multiple features:

- CGPA
- IQ
- gender encoding

**How it works**

- standardizes the input features first
- initializes all weights to zero
- in each epoch, uses the **entire dataset** to compute the gradient
- updates all weights and the bias once per epoch
- stores loss history so you can inspect learning progress

Batch gradient descent means:

- every update is based on all training rows
- updates are smoother and more stable
- training is usually easier to reason about

**Why use it**

- good when the dataset is small or medium sized
- stable learning behavior
- easier to debug than SGD
- good for understanding vectorized multi-feature regression

**Tradeoffs**

- slower on large datasets because every update uses every sample
- needs feature scaling for reliable convergence

### 3. `StochasticGradientDescentRegressor`

File: `stochastic_gradient_descent_example.py`

**What it does**

This model solves the same salary-style regression problem, but it updates the model using **one sample at a time**.

**How it works**

- standardizes the features
- shuffles the data each epoch
- picks one row
- predicts for that row only
- computes the gradient from that single example
- updates weights immediately
- repeats across all rows and epochs

This is stochastic gradient descent, often called **SGD**.

**Why use it**

- faster updates
- useful when data is very large
- works well for streaming or online learning ideas
- often reaches a good solution without waiting for full-dataset updates

**Tradeoffs**

- training is noisier
- loss may bounce around more
- usually needs more epochs and careful learning-rate tuning

### 4. `LinearRegression` from scikit-learn

Files:

- `by_sk_learn.ipynb`
- `batch_gradient_decent_for_diabetes.ipynb.ipynb`
- `stochastic_for_diabetes.ipynb`

**What it does**

This is the library baseline used to compare your custom implementations against a trusted, optimized regression model.

**How it works**

`LinearRegression` solves linear regression using optimized numerical methods from scikit-learn instead of your manual update loops.

You call:

- `fit()` to train
- `predict()` to infer values
- metrics like `r2_score` or `mean_squared_error` to evaluate it

**Why use it**

- fast and reliable baseline
- great for checking whether your custom implementation is behaving correctly
- simpler for real projects where you want a tested library solution

**Why keep the custom models too**

Because the custom models teach:

- gradient computation
- convergence behavior
- the difference between batch and stochastic updates
- why scaling and learning rate matter

## Batch GD vs SGD

| Topic | Batch Gradient Descent | Stochastic Gradient Descent |
| --- | --- | --- |
| Update timing | Once per epoch | Once per sample |
| Gradient source | Full dataset | Single sample |
| Training behavior | Smooth and stable | Noisy and jumpy |
| Speed per update | Slower | Faster |
| Best for | Small/medium datasets | Large datasets / online learning |
| Tuning difficulty | Easier | Usually harder |

## Which Model Should You Use?

Use `GDregressor` in `main.py` when:

- you want to learn the math of slope and intercept updates
- you want the smallest possible example

Use `BatchGradientDescentRegressor` when:

- you want a clear multi-feature implementation
- you want stable training behavior
- your dataset is not huge

Use `StochasticGradientDescentRegressor` when:

- you want to learn sample-by-sample updates
- you care about scaling to larger datasets
- you want to understand noisy but efficient optimization

Use `LinearRegression` from scikit-learn when:

- you need a dependable baseline
- you want the fastest way to build a regression model
- you want to compare your manual code against a trusted implementation

## Project Files

- `main.py`: simple one-feature gradient descent regression
- `batch_gradient_descent_salary_example.py`: multi-feature batch gradient descent example
- `stochastic_gradient_descent_example.py`: multi-feature stochastic gradient descent example
- `by_sk_learn.ipynb`: learning notebook comparing scikit-learn and custom work
- `batch_gradient_decent_for_diabetes.ipynb.ipynb`: batch gradient descent on the diabetes dataset
- `stochastic_for_diabetes.ipynb`: stochastic gradient descent on the diabetes dataset

## How To Run

If you are using the project environment:

```bash
uv sync
```

Then run any script:

```bash
python main.py
python batch_gradient_descent_salary_example.py
python stochastic_gradient_descent_example.py
```

## What You Learn From This Repo

This repo is useful because it shows the same machine learning idea from different levels:

- beginner level: understand slope, bias, and loss
- intermediate level: move to multiple features and scaled inputs
- practical level: compare custom code with scikit-learn

If your main goal is learning, start with `main.py`, then read the batch example, then compare it with the stochastic version.
