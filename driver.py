import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans
import pca


def load_data():
    from numpy import genfromtxt
    return genfromtxt('audio_data.csv', delimiter=',')


def main():
    data = load_data()
    results = []
    np.random.seed(10)

    # pca_data = pca.pca(data, 2)[0]    #pca from scratch
    # pca_data = pca.pca_s(data, 2)     #pca from sk_learn library

    # code for simple run where k=2
    # k=2
    # random_centroids = np.random.randint(0, 128, k)
    # km = KMeans(k)
    # km.fit(data, random_centroids)

    for k in range(2, 11):
        random_centroids = np.random.randint(0, 128, k)
        km = KMeans(k)
        results.append(km.fit(data, random_centroids))          #comment this for without pca
        # results.append(km.fit(pca_data, random_centroids))    #comment this for with pca
    plt.plot(results, list(range(2, 11)))
    plt.show()
    plt.savefig('k_means.png')


if __name__ == '__main__':
    main()