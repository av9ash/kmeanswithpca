from numpy import dot, mean, std, empty, argsort
from numpy.linalg import eigh
from sklearn.decomposition import PCA


def cov(data):
    # return dot(X.T, X) / X.shape[0]
    N = data.shape[1]
    C = empty((N, N))
    for j in range(N):
      C[j, j] = mean(data[:, j] * data[:, j])
      for k in range(j + 1, N):
          C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    return C


def pca(data, pc_count = None):
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V


def pca_s(data, pc_count = None):
    return PCA(n_components = 4).fit_transform(data)

# trans = pca(data, 2)[0]
# trans2 = pca2(data2, 2)
