
import torch
from torch import nn
import numpy as np
from .base_model import BaseModel



class LSTM_AE(BaseModel):
    def __init__(self, name, train_iters=10, hidden_size=64, layer_dim=3 ):
        self.name = name
        self.train_iters = train_iters
        self.hidden_size = hidden_size
        self.layer_dim = layer_dim


    def predict(self, X, **kwargs):
        X_true, X = self.lag_transform(X)
        X =  torch.from_numpy(X).float()
        X = X.unsqueeze(0) # batch dim
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy() # should cnversion be handled here?


    def fit_transform(self,X):
        self.fit(X)
        return self.predict(X)
        

    def fit(self,X,**kwargs):
        lag_size = 20
        X_true, X = self.lag_transform(X)
        X_true, X = torch.from_numpy(X_true), torch.from_numpy(X)
        X_true, X = X_true.float(), X.float()
        X =  X.unsqueeze(0) # TODO: later this would be fixed by dataloader?
        # ditch using timesteps and use previous data as input

        # TODO: fix initialization
        self.input_size = X.shape[1]
        self.model = RecurrentAutoencoder(self.X.shape[1], self.input_size, self.hidden_size)
        # set optimizer
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}, ],
                                        lr=.1)
        
        loss_fxn = torch.nn.L1Loss(reduction='sum')#torch.nn.MSELoss()

        for i in range(self.train_iters):
            # zero gradients from previous step
            optimizer.zero_grad()
            # output from model
            output = self.model(X)
            # calc loss and backprop gradients
            loss = loss_fxn(output, y)
            loss.backward()
            optimizer.step()
            print(loss.item(), "Loss")
        

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x):
        # x = x.repeat(self.seq_len, self.n_features)
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
        )

    def forward(self, x):

        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape(1, self.embedding_dim)

        # return hidden_n.reshape((self.n_features, self.embedding_dim))


