import os
import torch
import time
import numpy as np
from .base_model import BaseModel
from torch.utils.data import DataLoader


class RNN(torch.nn.Module):
    '''
    https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    '''
    def __init__(self, input_size, batch_size, seq_len, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_len =  seq_len
        #Defining the layers
        # RNN Layer
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(self.batch_size, self.seq_len, self.hidden_dim)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class VanillaRNN(BaseModel):
    def __init__(self, name, train_iters=10, num_layers=3, hidden_size=1, batch_size=5, learning_rate=.01):
        super(VanillaRNN, self).__init__(name)
        self.name = name
        self.train_iters = train_iters
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def fit_transform(self,X,**kwargs):
        self.fit(X)
        return self.predict(X)

    def predict(self, X, **kwargs):
        self.multivariate = X.shape[1] > 1
        if not self.multivariate:
            X_true, X = self.lag_transform(X, lag=self.batch_size, horizon=self.horizon)
        X = torch.from_numpy(X).float()
        X =  self.batch_shape(X, self.batch_size)

        test_loader = DataLoader(X,
                                  batch_size = self.batch_size,
                                  shuffle = False,
                                  drop_last=True)
        predictions = []
        with torch.no_grad():
            for t, x in enumerate(test_loader):
                output, hidden_size = self.model(x)
                predictions.append(output)
        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.permute(1, 0, 2) # put batch in the middle
        predictions = self.batch_reshape(predictions.numpy())
        return predictions # should cnversion be handled here?



    def fit(self,X, **kwargs):
        # transform X here
        orig_shape = X.shape
        lag_size = 20
        self.multivariate = X.shape[1] > 1
        if not self.multivariate:
            X_true, X = self.lag_transform(X, lag=self.batch_size, horizon=self.horizon)
        else: # we are passed multple features so just reconstruct those instead of lags
            X_true =  X
        X_true, X = torch.from_numpy(X_true), torch.from_numpy(X)
        X_true, X = X_true.float(), X.float()
        X =  self.batch_shape(X, self.batch_size)
        X_true = self.batch_shape(X_true, self.batch_size)
        seq_len, n_features = X.shape[-2], X.shape[-1]

        self.input_size = X.shape[2]
        self.output_size =  X_true.shape[-1] # number of output features

        dataset = torch.utils.data.TensorDataset(X, X_true)
        train_loader = DataLoader(dataset,
                                  batch_size = self.batch_size,
                                  shuffle = False,
                                  drop_last=True)


        self.model = RNN(self.input_size, self.batch_size, seq_len,  self.output_size, self.hidden_size, self.num_layers)
        # set optimizer
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}, ],
                                     lr=self.learning_rate)
        loss_fxn = torch.nn.MSELoss()

        for i in range(self.train_iters):
            losses = []
            for  t, (x,x_true) in enumerate(train_loader):
                # zero gradients from previous step
                optimizer.zero_grad()
                # output from model
                output, hidden_state = self.model(x)
                # calc loss and backprop gradients
                loss = loss_fxn(output, x_true)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            print(np.mean(losses), "RNN Loss")
            self.loss.append(np.mean(losses))

