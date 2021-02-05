import numpy as np
import pandas as pd

class ELM:
    def __init__(self, neurons, n):
        self.weights =  np.matrix(np.random.uniform(-0.5, 0.5, (neurons, n)))
        self.bias = np.matrix(np.random.uniform(0, 1, (1, neurons)))

    def train(self, inputs, outputs):
        n = inputs.shape[1]
        N = inputs.shape[0]
        
        Z =  inputs.dot(self.weights.T) + self.bias
        H = np.tanh(Z)
        W = np.linalg.pinv(H).dot(outputs)
        return dict(weights=self.weights, bias=self.bias, Z=Z, W=W, H=H)    
    def predict(self, inputs, W):
        Z =  inputs.dot(self.weights.T) + self.bias
        H = np.tanh(Z)
        Y = H.dot(W)
        return np.sign(Y)
        
