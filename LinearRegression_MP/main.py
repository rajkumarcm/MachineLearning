import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re

# 3. Naturally convert them to float treating float as native dtype
def replace_non_numeric(x, mean):
    if not re.fullmatch('\d+\.*\d*', x):
        return mean
    else:
        return x

def get_valid_rows(series):
    b_mask = series.apply(lambda x: True if re.fullmatch('\d+\.*\d*', x) else False)
    return series[b_mask]

# 5. Loss function
def mse(y, pred):
    diff = y - pred
    return 1/y.shape[0] * (diff.T @ diff)[0][0]

# 6. Inferencing
def predict(X, W, b, p_idx):
    result = {}
    result[p_idx] = X @ W + b
    return result

def get_gradient(X, y, pred):
    y = np.reshape(y, [-1, 1])
    pred = np.reshape(pred, [-1, 1])
    return X.T @ (y - pred), (y - pred)


if __name__ == '__main__':
    # 1. Load the dataset
    df = pd.read_csv('data/auto-mpg.data', header=None, sep='\s+',
                     names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                            'year', 'origin', 'name'],
                     skip_blank_lines=True, on_bad_lines='warn')

    # 2. Clean the data
    col_types = df.dtypes
    cols_to_fix = []
    for col in df.columns:
        if not (np.array([col_types[col]]) == np.array([np.int32, np.int64, np.float32, np.float64])).any():
            cols_to_fix.append(col)

    for col in cols_to_fix:
        series = get_valid_rows(df.loc[:, col])
        series = series.astype(np.float32)
        series_mean = series.mean()
        df.loc[:, col] = df.loc[:, col].apply(lambda x: replace_non_numeric(x, series_mean))
        df.loc[:, col] = df.loc[:, col].astype(np.float32)

    # 3. Split the data
    n = df.shape[0]
    split_size = 2/3
    training_split = int(n * split_size)
    y = df.mpg
    df = df.drop(columns=['mpg', 'name'])
    X_train = df.iloc[:training_split]
    y_train = y.iloc[:training_split]
    X_val = df.iloc[training_split:]
    y_val = y.iloc[training_split:]

    # 3. Standardise data
    X_mean = X_train.mean(axis=0)
    X_sd = X_train.std(axis=0)
    y_mean = y.mean()
    y_sd = y.std()

    X_train = (X_train - X_mean)/X_sd
    X_val = (X_val - X_mean)/X_sd
    y_train = (y_train - y_mean)/y_sd
    y_val = (y_val - y_mean)/y_sd

    # 4. Correlation plot to avoid multicollinearity
    plt.figure(figsize=(9, 6))
    sns.heatmap(df.corr(method='pearson'), linewidth=0.5)
    plt.show()

    # 7. Define model
    np.random.seed(1234)
    W = np.random.random([df.shape[1], 1])
    b = np.random.random([1, 1])
    lr = 1e-1
    epochs = 51
    y_train = np.reshape(y_train.values, [-1, 1])
    y_val = np.reshape(y_val.values, [-1, 1])

    # Training---------------------------------------------------

    tr_losses = np.zeros([epochs])
    vl_losses = np.zeros([epochs])
    for epoch in range(epochs):
        n_processes = 1
        process_size = X_train.shape[0]//n_processes
        processes = []
        w_gradient = None
        b_gradient = None
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
                p = executor.submit(predict, X=x_subset, W=W, b=b, p_idx=p_idx)
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
                p = executor.submit(get_gradient, X=x_subset, y=y_subset, pred=pred_subset)
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
            W -= lr * w_gradient
            b -= lr * np.mean(b_gradient, axis=0)

            # Loss------------------------------------------------------------------------------------------------------
            tr_loss = mse(y_train, pred)
            tr_losses[epoch] = tr_loss
            print(f'Epoch: {epoch} Loss: {tr_loss}')

    plt.figure()
    sns.lineplot(list(range(51)), tr_loss, markers=True)
    plt.show()