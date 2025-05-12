import numpy as np

def binary_crossentropy_loss(y_true, y_pred):
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.random.randn(n_neurons)
    
    def forward(self, x):
        self.input = x 
        self.z = np.dot(self.weights, x) + self.biases
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, d_output, learning_rate):
        sigmoid_derivative = self.output * (1 - self.output)
        
        d_z = d_output * sigmoid_derivative
        d_weights = np.outer(d_z, self.input)  
        d_biases = d_z

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
     
        return np.dot(self.weights.T, d_z)
    

class NeuralNetwork:
    def __init__(self):
        self.layer1 = Layer(n_inputs=2, n_neurons=5)  
        self.layer2 = Layer(n_inputs=5, n_neurons=1)  

    def forward(self, x):
        out1 = self.layer1.forward(x)
        out2 = self.layer2.forward(out1)
        return out2
    
    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train):
                output = self.forward(x)
        
                loss = binary_crossentropy_loss(y, output)
                total_loss += loss
                
                d_output = binary_crossentropy_loss_derivative(y, output)
                d_output = self.layer2.backward(d_output, learning_rate)
                self.layer1.backward(d_output, learning_rate)
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train)}")
