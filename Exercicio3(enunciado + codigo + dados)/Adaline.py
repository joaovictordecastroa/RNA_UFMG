import numpy as np

class Adaline:
    def __init__(self, lenght, eta = 0.001, tolerance = 0.0001, max_epochs = 10000):
        self.eta = eta
        self.max_epochs = max_epochs
        self.epoch = 0
        self.tolerance = tolerance
        self.errors_per_epoch = []
        self.lenght = lenght
        self.weights = np.random.rand(lenght + 1)
        self.erro = 1


    def train(self, x, y):
        n = x.shape[0]
        mean_square_error = 1
        while ((self.epoch < self.max_epochs) & (mean_square_error >= self.tolerance)):
            test = self.test(x, y)
            mean_square_error = test['mean_square_error']
            self.erro = test['error']
            self.errors_per_epoch.append(mean_square_error)
            # delta rule
            self.weights[1:] += self.eta * x.T.dot(self.erro)
            self.weights[0] += self.eta * sum(self.erro)
            self.epoch += 1
        print('Eita')
        return mean_square_error


    def test(self, x, y):
        n = x.shape[0]
        output = self.activation_function(x)
        error = y - output
        mean_square_error = sum(error ** 2) / n
        return {'mean_square_error': mean_square_error, 'error': error}

    def activation_function(self, X):
        if (len(X.shape) >= 2):
            return X.dot(self.weights[1:]) + self.weights[0]
        return X * self.weights[1:] + self.weights[0]

