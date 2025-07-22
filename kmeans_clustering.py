import numpy as np


class KMeans:
    def __init__(self, n_cluster=8, max_iter=300, init="random", n_init=10):

        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.centroids = None

    def fit(self, X):
        self.centroids = self.init_centroids(X)

        for _ in range(self.max_iter):

            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            self.centroids = self.move_centroids(X, cluster_group)

            if (self.centroids == old_centroids).all():
                break
        # print(old_centroids)
        # print(self.inertia(X, cluster_group))
        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        distances = []
        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))

            index_pos = distances.index(min(distances))
            cluster_group.append(index_pos)
            distances.clear()
        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))
        return np.array(new_centroids)

    def predict(self, X):
        pass

    def fit_predict(self, X):
        pass

    def kmeans_plus_plus(self, X):
        pass

    def init_centroids(self, X):
        if self.init == "random":
            random_indices = np.random.choice(
                len(X), size=self.n_cluster, replace=False
            )
            return X[random_indices]
