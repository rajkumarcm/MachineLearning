import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import progressbar

class LReLU:
    alpha = 0.05
    def __call__(self, X):
        return np.max([self.alpha*X, X], axis=0) # a time-being fix

    def grad(self, X):
        D = np.copy(X)
        D[X > 0] = 1
        D[X < 0] = self.alpha
        return D

class MSE:
    def __call__(self, y_pred, y):
        diff = y_pred - y
        return np.ravel(1/len(y_pred) * diff.T @ diff)[0]

    def grad(self, y_pred, y):
        return (2/len(y_pred)) * (y_pred - y)

class Dense: # For now let's implement just Dense

    def __init__(self, input_units, units, Activation, reg='None', alpha=0.5, beta=0.5):
        """
        Feedforward layer
        :param input_units: input shape
        :param units:
        :param Act:
        :param cache_X: Boolean representing whether to cache the output from previous layer or input given this is
        the first layer
        """
        self.input_units = input_units
        self.output_units = units
        self.W = self.initialise_weights()
        self.activation = Activation()
        self.reg = reg
        self.alpha = alpha
        self.beta = beta

    def initialise_weights(self):
        W = np.random.normal(loc=0, scale=1, size=[self.input_units, self.output_units])
        W_Z = (W - np.mean(W))/np.std(W)

        while (W_Z < -2).any() or (W_Z > 2).any():
            indices = (W_Z < -2) | (W_Z > 2)
            resample_size = np.sum(indices)
            W[indices] = np.random.normal(loc=0, scale=1, size=[resample_size])
            W_Z = (W - np.mean(W)) / np.std(W)
        return W

    def __call__(self, X):
        try:
            H = X @ self.W
        except Exception:
            print('debug')
        return H, self.activation(H)

    # Just imagine we are working with caching enabled all the time.
    def __backward(self, incoming_grad, H_curr, Z_prev):
        # Use gradient descent for now
        tmp_grad = incoming_grad * self.activation.grad(H_curr)
        local_grad = Z_prev.T @ (tmp_grad)  # gradient for computing the weight-update
        # Inject gradient of regularization to local_grad-------------
        reg_grad = np.copy(self.W)
        l1_coeff = self.alpha
        l2_coeff = 2
        if self.reg == 'elastic':
            l1_coeff *= self.beta
            l2_coeff = (self.alpha * (1 - self.beta))/2
        if self.reg == 'l1' or self.reg == 'lasso' or self.reg == 'elastic':
            reg_grad[self.W >= 1] = l1_coeff
            reg_grad[self.W < 0] = -l1_coeff
            reg_grad[self.W == 0] = 0
        if self.reg == 'l2' or self.reg == 'ridge' or self.reg == 'elastic':
            if self.reg == 'elastic':
                reg_grad += l2_coeff * self.W
            else:
                reg_grad = l2_coeff * self.W
        local_grad += reg_grad
        #--------------------------------------------------------------
        outbound_grad = tmp_grad @ self.W.T  # gradient for the next layer
        return local_grad, outbound_grad

    def update_weights(self, incoming_grad, H_curr, Z_prev, lr):
        local_grad, outbound_grad = self.__backward(incoming_grad, H_curr, Z_prev)
        self.W -= lr * local_grad
        return outbound_grad

class Model:
    def __init__(self, config, Loss):
        """

        :param config: Similar to sequential in TF2.0
        """

        self.config = config
        # Keep this simple for now
        self.outputs = [None]*len(config)
        self.loss = Loss()

    def __forward(self, Z, training=True):
        # Need to do some checks to verify whether the shape of X match the configuration
        for i, layer in enumerate(self.config):
            H, Z = layer(Z)
            if training:
                self.outputs[i] = H
        return Z

    def __call__(self, X):
        return self.__forward(X)

    def validate(self, X, y, batch_size):
        steps = X.shape[0]//batch_size
        y_hat = np.zeros_like(y)
        for step in range(steps):
            si = step * batch_size
            ei = si + batch_size
            if step ==  steps-1:
                remaining_size = X.shape[0] - si
                ei = si + remaining_size

            batch_X = X[si:ei]
            tmp_yhat = self.__forward(batch_X, training=False)
            y_hat[si:ei] = tmp_yhat
        return self.loss(y_hat, y)

    def fit(self, X, y, X_val, y_val, epochs, batch_size, lr=1e-1, reg='None', alpha=0.5, beta=0.5, verbose=False):
        layer_refs = list(reversed(self.config))
        N = X.shape[0]
        steps = N//batch_size
        tr_loss = np.zeros([epochs])
        vl_loss = np.zeros([epochs])
        for epoch in range(epochs):
            for step in range(steps):
                si = step*batch_size
                ei = si + batch_size
                batch_X = X[si:ei]
                batch_y = y[si:ei]
                y_pred = self.__forward(batch_X)

                tr_loss[epoch] += self.loss(y_pred, batch_y)
                vl_loss[epoch] += self.validate(X_val, y_val, batch_size)

                # Inject magnitude of weights here for l1/l2 regularization
                # loop over all layers and sum all the absolute weights
                weight_mag = 0
                if reg == 'elastic':
                    beta = alpha * (1 - beta)
                    alpha *= beta

                if reg == 'l1' or reg == 'lasso' or reg == 'elastic':
                    for layer in layer_refs:
                        weight_mag += np.sum(np.abs(layer.W))
                    weight_mag += alpha * weight_mag

                if reg == 'l2' or reg == 'ridge' or reg == 'elastic':
                    for layer in layer_refs:
                        W = layer.W.ravel()
                        weight_mag += W.T @ W
                    weight_mag += beta * weight_mag

                tr_loss[epoch] += weight_mag
                vl_loss[epoch] += weight_mag

                incoming_grad = self.loss.grad(y_pred, batch_y)
                for i, layer in zip(range(len(layer_refs)-1, -1, -1), layer_refs):
                    if i == 0:
                        incoming_grad = layer.update_weights(incoming_grad=incoming_grad, H_curr=self.outputs[i],
                                         Z_prev=batch_X, lr=lr)
                    else:
                        incoming_grad = layer.update_weights(incoming_grad=incoming_grad, H_curr=self.outputs[i],
                                                             Z_prev=self.outputs[i-1], lr=lr)
            tr_loss[epoch] /= epochs
            vl_loss[epoch] /= epochs
            if verbose:
                print(f'Epoch {epoch} Loss: {np.round(np.mean(tr_loss[epoch]), 4)}, '
                      f'Validation Loss: {np.round(np.mean(vl_loss[epoch]), 4)}')
        return {'tr_loss':tr_loss, 'vl_loss':vl_loss}


if __name__ == '__main__':

    # np.random.seed(633)
    epochs = 120
    l1_epochs = l2_epochs = elastic_epochs = 120

    batch_size = 10
    lr = 1e-3
    # X = np.random.random([100, 5])
    # y = X @ np.random.random([5, 1])
    #
    # X_train = X[:50]
    # y_train = y[:50]
    #
    # X_val = X[50:]
    # y_val = y[50:]

    import pandas as pd
    df = pd.read_csv('data/insurance.csv')
    df.drop('region', 'columns', inplace=True)
    df.sex.replace({'female':1, 'male':0}, inplace=True)
    df.smoker.replace({'yes':0, 'no':1}, inplace=True)

    df -= df.mean(axis=0)
    df /= df.std(axis=0)

    X_train = df.iloc[:-100, :-1].to_numpy()
    y_train = df.iloc[:-100, -1].to_numpy()
    y_train = np.reshape(y_train, [-1, 1])

    X_val = df.iloc[-100:, :-1].to_numpy()
    y_val = df.iloc[-100:, -1].to_numpy()
    y_val = np.reshape(y_val, [-1, 1])

    l1_config = [Dense(X_train.shape[1], 5, LReLU, reg='l1'),
                 Dense(5, 1, LReLU, reg='l1')]
    l1_model = Model(l1_config, MSE)
    l1_hist = l1_model.fit(X_train, y_train, X_val, y_val, epochs=l1_epochs, batch_size=batch_size, lr=lr, reg='l1',
                           verbose=True)

    l2_config = [Dense(X_train.shape[1], 5, LReLU, reg='l2'),
                 Dense(5, 1, LReLU, reg='l2')]
    l2_model = Model(l2_config, MSE)
    l2_hist = l2_model.fit(X_train, y_train, X_val, y_val, epochs=l2_epochs, batch_size=batch_size, lr=lr, reg='l2')

    no_reg_config = [Dense(X_train.shape[1], 5, LReLU),
                     Dense(5, 1, LReLU)]
    model_no_reg = Model(no_reg_config, MSE)
    no_reg_hist = model_no_reg.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, lr=lr)

    elastic_config = [Dense(X_train.shape[1], 5, LReLU, reg='elastic'),
                      Dense(5, 1, LReLU, reg='elastic')]
    elastic_model = Model(elastic_config, MSE)
    elastic_hist = l2_model.fit(X_train, y_train, X_val, y_val, epochs=elastic_epochs, batch_size=batch_size,
                                lr=lr, reg='elastic')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(range(epochs), l1_hist['tr_loss'][:epochs], '-b', label='L1 reg tr loss')
    axes[0, 0].plot(range(epochs), l2_hist['tr_loss'][:epochs], '-r', label='L2 reg tr loss')
    axes[0, 0].plot(range(epochs), no_reg_hist['tr_loss'], '-g', label='No reg tr loss')
    axes[0, 0].plot(range(epochs), elastic_hist['tr_loss'][:epochs], '-m', label='Elastic net reg tr loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('MSE with regularization')
    axes[0, 0].set_title('Training loss with different reg schemes')
    axes[0, 0].legend()

    axes[0, 1].plot(range(l1_epochs), l1_hist['tr_loss'], '-b', label='L1 reg tr loss')
    axes[0, 1].plot(range(l1_epochs), l1_hist['vl_loss'], '-r', label='L1 reg vl loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('L1 regularization')
    axes[0, 1].legend()

    axes[1, 0].plot(range(l2_epochs), l2_hist['tr_loss'], '-b', label='L2 reg tr loss')
    axes[1, 0].plot(range(l2_epochs), l2_hist['vl_loss'], '-r', label='L2 reg vl loss')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('L2 regularization')
    axes[1, 0].legend()

    axes[1, 1].plot(range(epochs), no_reg_hist['tr_loss'], '-b', label='No reg tr loss')
    axes[1, 1].plot(range(epochs), no_reg_hist['vl_loss'], '-r', label='No reg vl loss')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('No regularization')
    axes[1, 1].legend()
    plt.show()

    plt.figure()
    plt.plot(range(elastic_epochs), elastic_hist['tr_loss'], '-b', label='Elastic net reg tr loss')
    plt.plot(range(elastic_epochs), elastic_hist['vl_loss'], '-r', label='Elastic net reg vl loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Elastic net regularization')
    plt.legend()
    plt.show()


