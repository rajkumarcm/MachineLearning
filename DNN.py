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
        return 1/len(y_pred) * diff.T @ diff

    def grad(self, y_pred, y):
        return (2/len(y_pred)) * (y_pred - y)

class Dense: # For now let's implement just Dense

    def __init__(self, input_units, units, Activation):
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
        H = X @ self.W
        return H, self.activation(H)

    # Just imagine we are working with caching enabled all the time.
    def __backward(self, incoming_grad, H_curr, Z_prev):
        # Use gradient descent for now
        tmp_grad = incoming_grad * self.activation.grad(H_curr)
        local_grad = Z_prev.T @ (tmp_grad)  # gradient for computing the weight-update
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
            batch_X = X[si:ei]
            y_hat[si:ei] = self.__forward(batch_X, training=False)
        return self.loss(y_hat, y)

    def fit(self, X, y, X_val, y_val, epochs, batch_size):
        lr = 1e-1
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
                incoming_grad = self.loss.grad(y_pred, batch_y)
                for i, layer in zip(range(len(layer_refs)-1, -1, -1), layer_refs):
                    if i == 0:
                        incoming_grad = layer.update_weights(incoming_grad=incoming_grad, H_curr=self.outputs[i],
                                         Z_prev=batch_X, lr=lr)
                    else:
                        incoming_grad = layer.update_weights(incoming_grad=incoming_grad, H_curr=self.outputs[-i],
                                                             Z_prev=self.outputs[i-1], lr=lr)
            tr_loss[epoch] /= epochs
            vl_loss[epoch] /= epochs
            print(f'Epoch {epoch} Loss: {np.round(np.mean(tr_loss[epoch]), 4)}, '
                  f'Validation Loss: {np.round(np.mean(vl_loss[epoch]), 4)}')

if __name__ == '__main__':
    X = np.random.random([100, 5])
    y = X @ np.random.random([5, 1])

    X_train = X[:50]
    y_train = y[:50]

    X_val = X[50:]
    y_val = y[50:]
    config = [Dense(5, 10, LReLU),
              Dense(10, 1, LReLU)]
    model = Model(config, MSE)
    model.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=10)




