from numpy import array
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

seq = [x for x in range(0, 1000, 10)]
# seq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# seq = [1, 53, 192, 74, 91, 114, 18, 91, 74]


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


sequence_length = 3
X, y = split_sequence(seq, sequence_length)
# for i in range(len(X)):
#     print(X[i], ' -> ', y[i])
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


class Preditor(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(Preditor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        h_0 = torch.zeros(self.n_layers, x.shape[0], self.hidden_size)
        c_0 = torch.zeros(self.n_layers, x.shape[0], self.hidden_size)
        output, (_, _) = self.lstm(x, (h_0, c_0))
        # print(f"LSTM {output.shape}")
        out = self.linear(output[:, -1, :])
        # print(f"Linear {out.shape}")
        return out.squeeze(1)


input_size = 1
batch_size = 1
hidden_size = 50
n_layers = 1


class ModelManager:
    def __init__(self, model, epochs, lr, loss_fn, opt, normalization=False):
        self.model = model
        self.epochs = epochs if not normalization else epochs//10
        self.lr = lr
        self.loss_fn = loss_fn
        self.opt = opt
        self.normalization = normalization

    def train(self, dataset, target):
        dataset_copy = dataset.clone()
        target_copy = target.clone()
        if self.normalization:
            dataset_copy /= 100
            target_copy /= 100

        losses = []
        # training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for i, seq in enumerate(dataset_copy):
                self.opt.zero_grad()
                y_hat = self.model(seq.view(batch_size, sequence_length, input_size))
                loss = self.loss_fn(y_hat, target_copy[i].reshape(1))
                loss.backward()
                total_loss += loss.item()
                self.opt.step()
                # print(f"Y: {target[i].tolist()}, y_hat: {y_hat.tolist()}")
            # print(f"Loss: {total_loss / len(dataset)} {[epoch+1]}")
            losses.append(total_loss/len(dataset))
        plt.plot(range(self.epochs), losses)
        plt.title(f"Epochs:{self.epochs}\nLR:{self.lr}")
        plt.show()
        return losses[-1]

    def test(self, sequence_test):
        sequence_test_copy = sequence_test.clone()
        if self.normalization:
            sequence_test_copy /= 100
        with torch.no_grad():
            predicted = self.model(sequence_test_copy)
        if self.normalization:
            predicted *= 100
        return predicted


x_test = array([[60, 70, 80],
                [70, 80, 90],
                [80, 90, 100]])
x_test = torch.from_numpy(x_test).float()
x_test2 = array([980, 990, 1000])
x_test2 = torch.from_numpy(x_test2).float()


combinations = [(500, 0.01), (500, 0.1),
                (200, 0.01), (200, 0.1), (200, 0.3),
                (100, 0.2), (100, 0.3), (100, 0.4),
                (50, 0.3), (50, 0.4)]
combinations_2 = [(1000, 0.001), (500, 0.01), (200, 0.01)]


print(f"Sequence: {x_test.tolist()}")
for epochs, lr in combinations_2:
    loss_fn = nn.MSELoss()
    model = Preditor(input_size, hidden_size, n_layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    model_manager = ModelManager(model, epochs, lr, loss_fn, opt, normalization=False)
    final_loss = model_manager.train(X, y)
    print(f"Loss: {final_loss} - Num epochs: {model_manager.epochs} - LR: {lr}")
    print(f"Predicted: {model_manager.test(x_test.view(x_test.shape[0], x_test.shape[1], 1)).tolist()}")

