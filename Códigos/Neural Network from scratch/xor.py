import numpy as np
from neural_network import NeuralNetwork, sigmoid, sigmoid_prime, MSELoss, MSELoss_prime
import matplotlib.pyplot as plt

lr = 0.3
nn = NeuralNetwork(2, lr)
nn.append_layer(2, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)
nn.append_layer(1, bias=1, activation=sigmoid, activation_prime=sigmoid_prime)

X = np.array([[0,0], [1, 0], [1, 1], [0, 1]])
y = np.array([0, 1, 0, 1])
losses = []

for epoch in range(2000):
    total_loss = 0
    for x, y0 in zip(X, y):
        predicted = nn.forward(x.reshape(1,-1))
        loss = MSELoss(predicted, y0)
        loss_derivative = MSELoss_prime(predicted, y0)
        nn.backward(loss_derivative)
        total_loss += loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1} - Loss: {total_loss}")
    losses.append(total_loss)

plt.plot(losses)
plt.show()