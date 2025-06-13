import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / np.sum(e_x)

def linear(x):
    return x

def sigmoid_derivative(output):
    return output * (1 - output)

def relu_derivative(output):
    return (output > 0).astype(float)

def softmax_derivative(output):
    return output * (1 - output)

def linear_derivative(output):
    return np.ones_like(output)

def binary_crossentropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true

def categorical_crossentropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

def categorical_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true  

def mean_squared_error(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def mean_squared_error_derivative(y_true, y_pred):
    return y_pred - y_true


class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros(n_neurons)
        self.activation_name = activation

        self.activation_funcs = {
            'sigmoid': sigmoid,
            'relu': relu,
            'softmax': softmax,
            'linear': linear
        }

        self.activation_derivatives = {
            'sigmoid': sigmoid_derivative,
            'relu': relu_derivative,
            'softmax': softmax_derivative,
            'linear': linear_derivative
        }

    def forward(self, x):
        self.input = x
        self.z = np.dot(self.weights, x) + self.biases
        self.output = self.activation_funcs[self.activation_name](self.z)
        return self.output

    def backward(self, d_output, learning_rate):
        d_z = d_output * self.activation_derivatives[self.activation_name](self.output)
        d_weights = np.outer(d_z, self.input)
        d_biases = d_z

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return np.dot(self.weights.T, d_z)

    

class NeuralNetwork:
    def __init__(self):
        self.l1 = Layer(784, 128, activation='relu')
        self.l2 = Layer(128, 64, activation='relu')
        self.l3 = Layer(64, 10, activation='softmax')

    def forward(self, x):
        out1 = self.l1.forward(x)
        out2 = self.l2.forward(out1)
        out3 = self.l3.forward(out2)
        return out3

    def train(self, X, Y, epochs, lr):
        for epoch in range(1, epochs + 1):
            loss_sum = 0
            for x, y in zip(X, Y):
                y_pred = self.forward(x)
                loss_sum += categorical_crossentropy(y, y_pred)
                d3 = categorical_crossentropy_derivative(y, y_pred)
                d2 = self.l3.backward(d3, lr)
                d1 = self.l2.backward(d2, lr)
                _ = self.l1.backward(d1, lr)
            
            if epoch % 2 == 0:
                avg_loss = loss_sum / len(X)
                print(f'Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}')

    def predict(self, X):
        out = [self.forward(x) for x in X]
        return np.array(out)