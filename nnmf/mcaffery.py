# nmf_demo.py

# non-negative matrix factorization
# based on:
# en.wikipedia.org/wiki/Non-negative_matrix_factorization

# https://jamesmccaffrey.wordpress.com/2024/07/23/exploring-non-negative-matrix-factorization-from-scratch-using-python/

import numpy as np

# -----------------------------------------------------------

def update_H(W, H, V):
    N = np.matmul(W.T, V)  # numerator
    D = np.matmul(np.matmul(W.T, W), H)  # denominator
    result = np.multiply(H, np.divide(N, D))
    return result

def update_W(W, H, V):
    N = np.matmul(V, H.T)  # numerator
    D = np.matmul(np.matmul(W, H), H.T)
    result = np.multiply(W, np.divide(N, D))
    return result

def nmf_compute(V, k, max_iter):
    rnd = np.random.RandomState(2)  # could pass seed as param
    (m,n) = V.shape
    W = rnd.random((m,k))
    H = rnd.random((k,n))
    for i in range(max_iter):
        H = update_H(W, H, V)
        W = update_W(W, H, V)
        if i % (max_iter // 5) == 0:
            err = error(W, H, V)
            print("iteration = %4d  error = %0.4f " % (i, err))
    return (W, H)

def error(W, H, V):
    # "Frobenius norm" of diff between V and W*H
    err = np.sqrt(np.sum((V - np.matmul(W,H))**2))
    return err

# -----------------------------------------------------------
# here's a version that's self-contained (no helpers)
# call like (W, H, err) =  nmf_compute2(V, 2, 10000, seed=1)
# -----------------------------------------------------------

def nmf_compute2(V, k, max_iter, seed):
    rnd = np.random.RandomState(seed)
    (m,n) = V.shape
    W = rnd.random((m,k))
    H = rnd.random((k,n))
    for i in range(max_iter):
        # H = update_H(W, H, V)
        N = np.matmul(W.T, V)  # numerator
        D = np.matmul(np.matmul(W.T, W), H)  # denominator
        H = np.multiply(H, np.divide(N, D))
        # W = update_W(W, H, V)
        N = np.matmul(V, H.T)  # numerator
        D = np.matmul(np.matmul(W, H), H.T)
        W = np.multiply(W, np.divide(N, D))

        if i % (max_iter // 5) == 0:
            # err = error(W, H, V)
            err = np.sqrt(np.sum((V - np.matmul(W,H))**2))
            print("iteration = %4d  error = %0.4f " % (i, err))
    err = np.sqrt(np.sum((V - np.matmul(W,H))**2))
    return (W, H, err)

# -----------------------------------------------------------

def main():
    print("\nBegin NMF from scratch demo ")
    np.random.seed(0)
    np.set_printoptions(precision=1, suppress=True,
                        floatmode='fixed')

    m = 4
    n = 8  # V is (m,n)
    k = 2  # W = (m,k), H = (k,n), V = W*H

    V = [[2.0,  4.0,  6.0,  8.0, 10.0],
         [4.0,  8.1, 12.0, 16.0, 19.9],
         [1.0,  1.9,  3.0,  4.0,  4.9],
         [5.0, 11.0, 17.1, 23.0, 29.0]]
    V = np.array(V)
    print("\nV = ")
    print(V)

    np.set_printoptions(precision=4)
    print("\nStart compute NMF from scratch ")
    (W, H) = nmf_compute(V, k, 10000)
    # (W, H, err) = nmf_compute2(V, k, 10000, seed=1)
    print("Done ")

    print("\nW = ")
    print(W)
    print("\nH = ")
    print(H)

    print("\nW * H = ")
    WH = np.matmul(W, H)
    print(WH)

    err = error(W, H, V)
    print("\nerror = %0.4f " % err)

    print("\nStart compute NMF using scikit ")
    from sklearn.decomposition import NMF
    model = NMF(n_components=2, max_iter=10000,
                init='random', random_state=2, tol=0.0)
    W = model.fit_transform(V)
    H = model.components_
    print("Done ")

    print("\nW = ")
    print(W)
    print("\nH = ")
    print(H)

    print("\nW * H = ")
    WH = np.matmul(W, H)
    print(WH)

    err = model.reconstruction_err_
    print("\nerror = %0.4f " % err)

    print("\nEnd demo ")

if __name__ == "__main__":
    main()