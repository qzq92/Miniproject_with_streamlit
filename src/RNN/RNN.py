import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class TrainLSTM:
    def __init__(self, input_size, hidden_size, num_layers, lr=0.01, batch_size=32, n_epochs=1000):
        self.model = LSTM(input_size, hidden_size, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    def train_model(self, X_train, y_train, X_test, y_test):
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=self.batch_size)
        for epoch in range(self.n_epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Validation
            if epoch % 100 != 0:
                continue
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_train)
                train_loss = self.loss_fn(y_pred, y_train)
                y_pred = self.model(X_test)
                test_loss = self.loss_fn(y_pred, y_test)
            print("Epoch %d: train loss %.4f, test loss %.4f" % (epoch, train_loss, test_loss))
        return self.model