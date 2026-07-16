
#%%
import numpy as np

def cholesky(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                s = A[i, i] - np.dot(L[i, :i], L[i, :i])
                if s < 0 and abs(s) < 1e-12:
                    s = 0.0
                elif s < 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                L[i, i] = np.sqrt(s)
            else:
                s = A[i, j] - np.dot(L[i, :j], L[j, :j])
                L[i, j] = s / L[j, j]

    return L

def solveLU(L, U, b):
    L = np.array(L, float)
    U = np.array(U, float)
    b = np.array(b, float)
    n = L.shape[1]
    # forward substitution - Ly = b
    y = np.zeros([n])
    for j in range(n):
        # sumk = 0
        # for k in range(j):
        #     sumk += L[j, k] * y[k]
        sumk = np.dot(L[j, :j], y[:j])
        y[j] = (b[j] - sumk)/L[j, j]
        
    #backward substitution - Lx = y
    x = np.zeros([n])
    for j in range(n-1, -1, -1):
        x[j] = (y[j] - np.dot(U[j, j+1:], x[j+1:]))/U[j, j]
    return x


if __name__ == "__main__":
    # H = np.array([[5.2, 3, 0.5, 1, 2],
    #               [3, 6.3, -2, 4, 0],
    #               [0.5, -2, 8, -3.1, 3],
    #               [1, 4, -3.1, 7.6, 2.6],
    #               [2, 0, 3, 2.6, 15]
    #               ])
    H = np.array([[8, 3.22, 0.8, 0, 4.1],
                  [3.22, 7.76, 2.33, 1.91, -1.03],
                  [0.8, 2.33, 5.25, 1, 3.02],
                  [0, 1.91, 1, 7.5, 1.03],
                  [4.1, -1.03, 3.02, 1.03, 6.44]])
    b = np.array([9.45, -12.2, 7.78, -8.1, 10.0])
    L = cholesky(H)
    print(L)
    b = np.array([9.45, -12.2, 7.78, -8.1, 10])
    x = solveLU(L, L.T, b)
    print(x)
# %%
