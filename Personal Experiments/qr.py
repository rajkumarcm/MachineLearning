#%%
import numpy as np

def norm(x):
    return np.sqrt(x.T @ x)

def householder_transformation(A):
    n, f = A.shape
    H_acc = []
    for j in range(f):
        x = A[j:, j]
        # e-> [1,0,0...]
        e1 = np.ones_like(x)
        e1[1:] = 0

        v = x - norm(x) * e1
        u = v/norm(v)
        u = u[:, np.newaxis]
        # Reflection matrix
        H = np.eye(len(x), len(x)) - 2 * u @ u.T
        # Embed H into a 3x3 matrix....
        embed_size = n - len(x)
        H_master = np.eye(n, n)
        H_master[embed_size:, embed_size:] = H
        H = H_master
        A = H @ A
        H_acc.append(H)

    R = A.copy()
    H_master = H_acc[0]
    for i in range(1, len(H_acc)):
        H_master = H_master @ H_acc[i]
    Q = H_master.copy()
    return Q, R

if __name__ == "__main__":
    A = np.array([[4, 1], [3, 2], [0, 5]])
    Q, R = householder_transformation(A)
    print(Q)
    print(R)
# %%
