import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base_model import BaseModel
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, number_of_features, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        self.number_of_features = number_of_features
        self.loss = []

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError
        
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)
        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError
        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class VRAE(nn.Module, BaseModel):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, name, batch_size=1, hidden_size=90, hidden_layer_depth=1, latent_length=20,
                 learning_rate=0.005, block='LSTM',  train_iters=5, dropout_rate=0.,
                 optimizer_name='Adam', loss_fn='MSELoss',cuda=False, print_every=100, clip=True,
                 max_grad_norm=5, dload='.'):
        super(VRAE, self).__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.learning_rate = learning_rate
        self.block = block
        self.train_iters = train_iters
        self.dropout_rate = dropout_rate
        self.optimizer_name = optimizer_name
        self.loss_fn = loss_fn
        self.loss = []
        self.print_every = print_every
        self.cuda = False # keep false for now
        self.max_grad_norm = max_grad_norm
        self.dload = dload
        self.is_fitted = False
        self.batch_size =  batch_size
        self.clip = clip
        self.multivariate = False

        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor


    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.train_iters,
                batch_size=self.batch_size,
                cuda=self.use_cuda)
    
    def build_model(self, sequence_length, number_of_features, output_size=1,batch_size=1):
        if self.use_cuda:
            self.cuda()
        print("SEQ LNE:", sequence_length)
        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=self.hidden_size,
                               hidden_layer_depth=self.hidden_layer_depth,
                               latent_length=self.latent_length,
                               dropout=self.dropout_rate,
                               block=self.block)

        self.lmbd = Lambda(hidden_size=self.hidden_size,
                           latent_length=self.latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = self.batch_size,
                               hidden_size=self.hidden_size,
                               hidden_layer_depth=self.hidden_layer_depth,
                               latent_length=self.latent_length,
                               output_size=output_size,
                               number_of_features = number_of_features,
                               block=self.block,
                               dtype=self.dtype)

        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            import pdb; pdb.set_trace()
            raise ValueError('Not a recognized optimizer')

        if self.optimizer == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif self.optimizer == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)
        else:
            self.loss_fn = nn.MSELoss(size_average=False)
        
        # this is the only class that does not have a wrapper
        # calls to self.model should be self referential
        # self.model = self
        # NOTE: they cannot be self referential b.c. call .train() causes recursion depth

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)
        
        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function
        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X, X_true):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration
        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)
        x_decoded, _ = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, X_true, self.loss_fn)#x.detach(), self.loss_fn)
        return loss, recon_loss, kl_loss, x


    def _train(self, train_loader):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times
        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()

        epoch_loss = 0
        t = 0
        for t, (X, X_true) in enumerate(train_loader):
            # Index first element of array to return tensor
            # X = X[0]
            # X_true = X_true[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)
            X_true = X_true.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, _ = self.compute_loss(X,X_true)
            loss.backward()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

            # accumulator
            epoch_loss += loss.item()

            self.optimizer.step()

            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item()))
        # normally this line is outside batch train loop
        # put back T later
        print('Average loss: {:.4f}'.format(epoch_loss / t))
        self.loss.append(np.mean(epoch_loss / t))


    def fit(self, X, save = False, **kwargs):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`
        :param dataset: `X`  np.array 
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """
        self.multivariate = X.shape[1] > 1
        if not self.multivariate:
            X_true, X = self.lag_transform(X, lag=self.batch_size, horizon=self.horizon)
        else: # we are passed multple features so just reconstruct those instead of lags
            X_true =  X
        X_true, X = torch.from_numpy(X_true), torch.from_numpy(X)
        X_true, X = X_true.float(), X.float()
        X =  self.batch_shape(X, self.batch_size)
        X_true = self.batch_shape(X_true, self.batch_size)
        #these are always the correct inds in batch v. nonbatch
        seq_len, n_features = X.shape[-2], X.shape[-1]

        if self.multivariate:
            # if we were passed multiple features then we don't use lags, just reconstruct features
            self.build_model(seq_len,n_features, batch_size=self.batch_size, output_size=n_features)
        else:
            self.build_model(seq_len,n_features, batch_size=self.batch_size)
        dataset = torch.utils.data.TensorDataset(X,X_true)

        train_loader = DataLoader(dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)
    
        for i in range(self.train_iters):
            print('Epoch: %s' % i)

            self._train(train_loader)

        self.is_fitted = True
        if save:
            self.save('model.pth')


    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function
        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function
        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, X, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """
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

        self.eval()
        dataset = torch.utils.data.TensorDataset(X)
        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False, # Don't shuffle for test_loader
                                 drop_last=True) 
        
        # X.shape[0] * X.shape[1] % btch_shape^2
        loader_dropped = X.shape[0] * X.shape[1]  % self.batch_size**2
        if loader_dropped != 0:
            self.trimmed +=  loader_dropped


        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                x_decoded = np.swapaxes(x_decoded,1,0) # swap axes back to our canoncial repr. (num_batches, batch_size, n_feats)
                x_decoded = self.batch_reshape(x_decoded)
                return x_decoded

        raise RuntimeError('Model needs to be fit')


    def predict(self, X, save = False, **kwargs):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        final_x = X
        lag_size = 20
        X_true, X = self.lag_transform(X, lag=self.batch_size, horizon=self.horizon)
        X_true, X = torch.from_numpy(X_true), torch.from_numpy(X)
        X_true, X = X_true.float(), X.float()

        X =  self.batch_shape(X, self.batch_size)
        X_true = self.batch_shape(X_true, self.batch_size)

        #these are always the correct inds in batch v. nonbatch
        seq_len, n_features = X.shape[-2], X.shape[-1]
        dataset = torch.utils.data.TensorDataset(X,X_true)


        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, (x, x_true) in enumerate(test_loader):
                    # x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False, **kwargs):
        """
        Combines the `fit` and `transform` functions above
        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.reconstruct(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))