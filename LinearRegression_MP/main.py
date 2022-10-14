"""
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Linear Regression Distributed Training
"""

import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

class LinearRegression_MP:

    def __init__(self, alpha=1, gamma=1, lr=1e-1):
        self.alpha = alpha
        self.gamma = gamma
        self.lr0 = lr
        np.random.seed(1234)

    # 3. Naturally convert them to float treating float as native dtype
    def replace_non_numeric(self, x, mean):
        if not re.fullmatch('\d+\.*\d*', x):
            return mean
        else:
            return x

    def get_valid_rows(self, series):
        b_mask = series.apply(lambda x: True if re.fullmatch('\d+\.*\d*', x) else False)
        return series[b_mask]

    # 5. Loss function
    def mse(self, y, pred):
        diff = y - pred
        return float(1/y.shape[0] * (diff.T @ diff)[0][0])

    # 6. Inferencing
    def predict(self, X, p_idx):
        result = {}
        result[p_idx] = X @ self.W + self.b
        return result

    def get_gradient(self, X, y, pred):
        y = np.reshape(y, [-1, 1])
        pred = np.reshape(pred, [-1, 1])
        w_grad = X.T @ (y - pred) - \
                 (self.alpha * self.gamma * np.sign(self.W)) - \
                 (self.alpha * (1 - self.gamma)) * self.W
        return w_grad, (y - pred) # for weights, and bias


    def fit(self, X_train, y_train, X_val, y_val, epochs=15, n_workers=16):

        # 7. Define model
        self.W = np.random.random([X_train.shape[1], 1])
        self.b = np.random.random([1, 1])

        y_train = np.reshape(y_train.values, [-1, 1])
        y_val = np.reshape(y_val.values, [-1, 1])

        # Training---------------------------------------------------

        # The following parameters for each model
        tr_losses = np.zeros([epochs])
        vl_losses = np.zeros([epochs])
        for epoch in range(epochs):
            n_processes = n_workers
            process_size = X_train.shape[0]//n_processes

            vl_process_size = X_val.shape[0]//n_processes
            processes = []
            w_gradient = None
            b_gradient = None

            # s = 10
            # self.lr = self.lr0 / (1 + (epoch / n_processes))
            if epoch <= 20:
                self.lr = 1e-1
            elif epoch > 20 and epoch <= 25:
                self.lr = 1e-2
            elif epoch > 25:
                self.lr = 1e-3

            # Inferencing
            with concurrent.futures.ProcessPoolExecutor() as executor:

                # Prediction------------------------------------------------------------------------------------------------
                pred = np.zeros([n_processes, process_size])
                for p_idx in range(n_processes):
                    start_idx = p_idx * process_size
                    _length = start_idx + process_size
                    if p_idx == n_processes-1:
                        _length = X_train.shape[0]
                    x_subset = X_train.iloc[start_idx:_length]
                    p = executor.submit(self.predict, X=x_subset,p_idx=p_idx)
                    processes.append(p)

                for process in concurrent.futures.as_completed(processes):
                    result = process.result()
                    p_idx = list(result.keys())[0]
                    result = pd.Series(result.values()).iloc[0].values
                    pred[p_idx, :_length] = result[:process_size, 0]

                    remaining = None
                    if p_idx == n_processes - 1:
                        pred = np.reshape(pred, [-1, 1])
                        remaining_size = X_train.shape[0] - int(process_size * n_processes)
                        if remaining_size > 0:
                            pred = list(pred.reshape([-1]))
                            pred.extend(result[-remaining_size:, 0])
                            pred = np.array(pred).reshape([-1, 1])

                # Val val_prediction------------------------------------------------------------------------------------------------
                val_pred = np.zeros([n_processes, vl_process_size])
                processes = []
                for p_idx in range(n_processes):
                    start_idx = p_idx * vl_process_size
                    _length = start_idx + vl_process_size
                    if p_idx == n_processes - 1:
                        _length = X_val.shape[0]
                    x_subset = X_val.iloc[start_idx:_length]
                    p = executor.submit(self.predict, X=x_subset, p_idx=p_idx)
                    processes.append(p)

                for process in concurrent.futures.as_completed(processes):
                    result = process.result()
                    p_idx = list(result.keys())[0]
                    result = pd.Series(result.values()).iloc[0].values
                    val_pred[p_idx, :_length] = result[:vl_process_size, 0]

                    if p_idx == n_processes - 1:
                        val_pred = np.reshape(val_pred, [-1, 1])
                        remaining_size = X_val.shape[0] - int(vl_process_size * n_processes)
                        if remaining_size > 0:
                            val_pred = list(val_pred.reshape([-1]))
                            val_pred.extend(result[-remaining_size:, 0])
                            val_pred = np.array(val_pred).reshape([-1, 1])

                # Compute gradient------------------------------------------------------------------------------------------
                processes = []
                for p_idx in range(n_processes):
                    start_idx = p_idx * process_size
                    _length = start_idx + process_size
                    if p_idx == n_processes-1:
                        _length = X_train.shape[0]
                    x_subset = X_train.iloc[start_idx:_length]
                    y_subset = y_train[start_idx:_length]
                    pred_subset = pred[start_idx:_length]
                    p = executor.submit(self.get_gradient, X=x_subset, y=y_subset, pred=pred_subset)
                    processes.append(p)

                for process in concurrent.futures.as_completed(processes):
                    [w_grad, b_grad] = process.result()
                    if w_gradient is None:
                        w_gradient = w_grad
                        b_gradient = np.sum(b_grad)[None, None]
                    else:
                        w_gradient += w_grad
                        b_gradient += np.sum(b_grad)[None, None]
                w_gradient = -2/X_train.shape[0] * w_gradient
                b_gradient = -2/X_train.shape[0] * b_gradient

                # Update weights--------------------------------------------------------------------------------------------
                self.W -= self.lr * w_gradient
                self.b -= self.lr * np.mean(b_gradient, axis=0)

                # Loss------------------------------------------------------------------------------------------------------
                tr_loss = self.mse(y_train, pred)
                vl_loss = self.mse(y_val, val_pred)
                tr_losses[epoch] = tr_loss
                vl_losses[epoch] = vl_loss
                print(f'Epoch: {epoch} Loss: {tr_loss} Val_Loss: {vl_loss}')
        return {'tr':tr_losses, 'vl':vl_losses}, tr_loss, vl_loss

# plt.figure()
# sns.lineplot(list(range(51)), tr_loss, markers=True)
# plt.show()

if __name__ == '__main__':
    X_train = np.load('data/X_train.npy')
    X_val = np.load('data/X_val.npy')
    y_train = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')

    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)

    eta_grid = [1e-3, 1e-2, 1e-1]
    alpha_grid = [0.01, 0.05, 0.1, 0.15, 0.2]
    gamma_grid = []

    lr_mp = LinearRegression_MP(alpha=params[1], gamma=params[2], lr=params[0])
    history, tr_loss, vl_loss = lr_mp.fit(X_train, y_train, X_val, y_val, epochs=50)

    plt.figure()
    plt.plot(list(range(50)), history['tr'], '-b', label='Training')
    plt.plot(list(range(50)), history['vl'], '-r', label='Validation')
    plt.legend()
    plt.show()