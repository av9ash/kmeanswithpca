import numpy as np


class KMeans:
    def __init__(self, k=2, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = None

    def fit(self, data, random_centroids):
        z = 0
        for i in random_centroids:
            self.centroids[z] = data[i]
            z+=1

        for i in range(self.max_iter):
            print(i)
            self.classifications = {}

            for j in range(self.k):
                self.classifications[j] = []

            for feature_set in data:
                distances = [np.linalg.norm(feature_set-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            opt = True
            for j in range(self.k):
                opt = np.array_equal(prev_centroids[j], self.centroids[j]) and opt

            if opt:
                break

        distance = 0
        for classification in self.classifications:
            x = self.classifications[classification]
            y = self.centroids[classification]

            for item in x:
                distance = (np.linalg.norm(item - y))**2 + distance

        return distance
