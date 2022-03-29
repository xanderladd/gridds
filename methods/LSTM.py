import os
import torch
import time
import numpy as np
from .base_model import BaseModel
from torch.utils.data import DataLoader



class LSTM(BaseModel):
    def __init__(self, name, batch_size=10, train_iters=10, \
    hidden_size=64, layer_dim=1, learning_rate=.2 ):
        super(LSTM, self).__init__(name)
        self.name = name
        # parameters
        self.train_iters = train_iters
        # TODO: fix initialization
        self.hidden_size = hidden_size
        self.layer_dim = layer_dim
        self.batch_size =  batch_size
        self.learning_rate =  learning_rate

        
    def predict(self, X, **kwargs):
        lag_size = 20
        self.multivariate = X.shape[1] > 1
        if not self.multivariate:
            X_true, X = self.lag_transform(X, lag=self.batch_size, horizon=self.horizon)
        X = torch.from_numpy(X).float()
        X =  self.batch_shape(X, self.batch_size)

        test_loader = DataLoader(X,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)
        predictions = []
        with torch.no_grad():
            for t, x in enumerate(test_loader):
                predictions.append(self.model(x))
        predictions = torch.cat(predictions, dim=1)
        predictions = predictions.permute(1, 0, 2) # put batch in the middle
        predictions = self.batch_reshape(predictions.numpy())
        return predictions # should cnversion be handled here?


    def fit_transform(self,X, **kwargs):
        self.fit(X)
        return self.predict(X)
        
    def fit(self,X,**kwargs):
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
                                  shuffle = True,
                                  drop_last=True)


        self.model = LSTM_Model(self.input_size, self.batch_size, seq_len, \
        self.layer_dim, self.output_size, self.hidden_size)
        # set optimizer
        optimizer = torch.optim.SGD([{'params': self.model.parameters()}, ],
                                     lr=self.learning_rate)
        loss_fxn = torch.nn.MSELoss()
        
        for i in range(self.train_iters): # was 50
            losses = []
            for  t, (x,x_true) in enumerate(train_loader):
                # zero gradients from previous step
                optimizer.zero_grad()
                # output from model
                output = self.model(x)
                # calc loss and backprop gradients
                loss = loss_fxn(output, x_true)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            print(np.mean(losses), "LSTM Loss")
            self.loss.append(np.mean(losses))
       
        return




class LSTM_Model(torch.nn.Module):
    '''
    https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
    '''
    def __init__(self, input_dim, batch_size, seq_len, layer_dim, output_dim, hidden_dim, dropout=.1):
        super(LSTM_Model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_len =  seq_len
        self.batch_size = batch_size

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True)#,  dropout=dropout)

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        
        out = out.contiguous().view(self.batch_size, self.seq_len, self.hidden_dim)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

    