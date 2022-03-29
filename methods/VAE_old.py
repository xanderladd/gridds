
import torch
from torch import nn
import numpy as np


class LSTM_VAE:
    def __init__(self, name, n_iters=10, alpha=10, gamma=1 ):
        self.name = name
        self.n_iters = n_iters
        self.alpha = alpha
        self.gamma = gamma

    def predict(self,site_data,**kwargs):
        # HACK : only using one ft.
        site_data = site_data[:,1:2]
        if len(site_data.shape) > 1:
            input_size = site_data.shape[1]
        else:
            input_size = 1
            
        # TODO: fix initialization
        hidden_size = 64
        output_size =  1
        batch_size = 1
        layer_dim = 3

        # simple sequential timesteps
        # X = torch.from_numpy(np.repeat(np.arange(len(site_data)).reshape(-1,1), input_size, axis=1)).unsqueeze(0).float()
        y = torch.from_numpy(site_data).float()#.unsqueeze(0)
        X = y.unsqueeze(0)
        # y = y[0,:,0]
        
        X = torch.stack([y[i-20:i] for i in range(20,len(y))])
        X = X.unsqueeze(0)
        y = y[20:].unsqueeze(-1) # for delay
        # y = torch.nn.functional.normalize(y, p=2.0, dim=1)
        # X = torch.nn.functional.normalize(X, p=2.0, dim=1)
        # TODO: hardcoded seq ln
        model = VAE(X.shape[1], 20, 64, 2)
        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=.05)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        loss_fxn = torch.nn.MSELoss()
        
        for i in range(50):
            # zero gradients from previous step
            optimizer.zero_grad()
            # output from model
            output, z, mu, std = model(X)
            # calc loss and backprop gradients
              # reconstruction loss
            loss = loss_fxn(output,y)#self.gaussian_likelihood(output, self.log_scale, y)

            # kl
            kl = self.kl_divergence(z, mu, std)

            # elbo
            elbo = (kl - loss)
            elbo = elbo.mean()
            elbo.backward()
            optimizer.step()

            print(elbo.item(), "ELBO")


        with torch.no_grad():
            res, z_pred, mu, std  = model(X)
        # HACK: res has only 4 dims
        res = res[:,:4]
        return res.numpy()



    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))#, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl



class VAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, latent_dim=5):
        super(RecurrentVAE, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, latent_dim, n_features)
        
        # distribution parameters
        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_var = nn.Linear(embedding_dim, latent_dim)
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)
        #return latent sample 
        # TODO:allow for more latent sample
        return x_hat, z, mu, std

'''

Take a stack of 40/50 time points so latent space is latent_dim x 40
40 (stack) x 4 (features) -> R^5

'''
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
        # x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.input_dim))

        # x = x.reshape((1, self.seq_len, self.input_dim))
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


