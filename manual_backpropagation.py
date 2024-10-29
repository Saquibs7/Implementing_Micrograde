import numpy as np
import pandas as pd

#setting the MLP Structure
class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.input = input  # Store input for use in backward pass
        self.output = input.dot(self.weights) + self.biases  # Store output for accuracy calculation
        return self.output

    def backward(self, grad_output):
        # Compute gradients for weights, biases, and input
        self.grad_weights = self.input.T.dot(grad_output)
        self.grad_biases = np.sum(grad_output, axis=0)
        grad_input = grad_output.dot(self.weights.T)
        return grad_input

    # Implementing batch Normalization
    # Normalize the inputs to have zero mean and unit variance, making training more stable and efficient
    class BatchNormalization:
        def __init__(self, dim, epsilon=1e-5):
            self.gamma = np.ones(dim)
            self.beta = np.zeros(dim)
            self.epsilon = epsilon

        def forward(self, input):
            # Store input for use in backward pass
            self.input = input

            # Calculate mean and variance for normalization
            self.mean = np.mean(input, axis=0)
            self.var = np.var(input, axis=0)

            # Normalize input
            self.x_normalized = (input - self.mean) / np.sqrt(self.var + self.epsilon)

            # Scale and shift
            output = self.gamma * self.x_normalized + self.beta
            return output

        def backward(self, grad_output):
            # Get batch size
            N = self.input.shape[0]

            # Compute gradients with respect to gamma and beta
            self.grad_gamma = np.sum(grad_output * self.x_normalized, axis=0)
            self.grad_beta = np.sum(grad_output, axis=0)

            # Backpropagate through normalization
            dx_normalized = grad_output * self.gamma
            dvar = np.sum(dx_normalized * (self.input - self.mean) * -0.5 * np.power(self.var + self.epsilon, -1.5),
                          axis=0)
            dmean = np.sum(dx_normalized * -1.0 / np.sqrt(self.var + self.epsilon), axis=0) + dvar * np.mean(
                -2.0 * (self.input - self.mean), axis=0)

            dx = dx_normalized / np.sqrt(self.var + self.epsilon) + dvar * 2.0 * (
                        self.input - self.mean) / N + dmean / N
            return dx

    # implementing the activation function
    class TanhActivation:
        def forward(self, x):
            self.output = np.tanh(x)
            return self.output

        def backward(self, grad_output):
            # Derivative of tanh is 1 - tanh^2(x)
            return grad_output * (1 - self.output ** 2)
# calculating cross entropy loss function
class CrossEntropyLoss:
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        self.predictions = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        self.predictions /= np.sum(self.predictions, axis=1, keepdims=True)
        # Compute the loss
        N = targets.shape[0]
        self.targets = targets
        loss = -np.sum(targets * np.log(self.predictions + 1e-15)) / N
        return loss

    def backward(self):
        # Gradient of cross-entropy loss with softmax
        N = self.targets.shape[0]
        grad_output = (self.predictions - self.targets) / N
        return grad_output
# asseembling everything to make two layer MLP to implement forward and backward pass
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Define the layers
        self.layer1 = LinearLayer(input_size, hidden_size)
        self.batchnorm1 = BatchNormalization(hidden_size)
        self.activation1 = TanhActivation()
        self.layer2 = LinearLayer(hidden_size, output_size)
        self.batchnorm2 = BatchNormalization(output_size)
        self.loss = CrossEntropyLoss()

    def forward(self, x, targets):
        # Forward pass
        out = self.layer1.forward(x)
        out = self.batchnorm1.forward(out)
        out = self.activation1.forward(out)
        out = self.layer2.forward(out)
        out = self.batchnorm2.forward(out)
        loss = self.loss.forward(out, targets)
        return loss

    def backward(self):
        # Backward pass
        grad_output = self.loss.backward()
        grad_output = self.batchnorm2.backward(grad_output)
        grad_output = self.layer2.backward(grad_output)
        grad_output = self.activation1.backward(grad_output)
        grad_output = self.batchnorm1.backward(grad_output)
        grad_output = self.layer1.backward(grad_output)
# testing the MLP
# Sample data
np.random.seed(0)
X = np.random.randn(10, 3)  # 10 samples, 3 features
y = np.zeros((10, 2))       # Binary classification (2 output classes)
y[np.arange(10), np.random.randint(0, 2, size=10)] = 1  # One-hot encoding for targets

# Define the accuracy function
def calculate_accuracy(predictions, targets):
    pred_labels = np.argmax(predictions, axis=1)  # Class with highest probability
    true_labels = np.argmax(targets, axis=1)      # True labels
    accuracy = np.mean(pred_labels == true_labels) * 100  # Percentage accuracy
    return accuracy

# Create the MLP
mlp = MLP(input_size=3, hidden_size=5, output_size=2)

# Forward pass
loss = mlp.forward(X, y)

# Calculate predictions and accuracy
predictions = mlp.layer2.output  # Output layer activations
accuracy = calculate_accuracy(predictions, y)

# Backward pass
mlp.backward()

# Output the loss, gradients, and accuracy
print("Loss:", loss)
print("Gradient of first layer weights:\n", mlp.layer1.grad_weights)
print("Accuracy:", accuracy, "%")
#we can clearly see that accuracy is 60 % becuase of following reasons
#1. we are using very less data set that is of 10 rows
#2. we go through only 1 epoch
#3. not using optimization algorithm like gradient descent

# increasing the number of epochs
# Number of epochs to train the model
num_epochs = 10

learning_rate = 0.01  # Define a learning rate

for epoch in range(num_epochs):
    # Forward pass
    loss = mlp.forward(X, y)

    # Calculate predictions and accuracy for monitoring
    predictions = mlp.layer2.output
    accuracy = calculate_accuracy(predictions, y)

    # Backward pass
    mlp.backward()

    # Gradient Descent Step (parameter update)
    # For each layer, we update weights and biases with the gradients
    mlp.layer1.weights -= learning_rate * mlp.layer1.grad_weights
    mlp.layer1.biases -= learning_rate * mlp.layer1.grad_biases
    mlp.layer2.weights -= learning_rate * mlp.layer2.grad_weights
    mlp.layer2.biases -= learning_rate * mlp.layer2.grad_biases

    # Print loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Accuracy: {accuracy}%")
