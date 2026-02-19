from sklearn.datasets import make_regression

import matplotlib.pyplot as plt 
import numpy as np 


X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20)


plt.scatter(X,y)

class GDregressor:
    def __init__(self, learning_rate= 0.001, epochs = 1000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 68.18 # hard coded value from ipynb file, can be calculated by hand or by sk learn
        self.b = 120;
        
    def fit(self, X, y):
    # calculate the b using GD
        for i in range(self.epochs):
            lose_slope = -2 * np.sum(y - self.m * X.ravel() - self.b)
            self.b = self.b - (self.learning_rate * lose_slope)
        
        print(self.b)
        
    
gd = GDregressor(0.001, 10000)
gd.fit(X,y)