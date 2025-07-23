import numpy as np


class KMeans:
    def __init__(
        self, n_clusters=8, max_iter=300, init="random", n_init=10, tol=0.0001
    ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.tol = tol
        self.centroids = None
        self.inertia_ = np.inf
        self.inertia_path_ = []
        self.centroids_path_ = []

    def _reset_variables(self):
        self.centroids = None
        self.inertia_ = np.inf
        self.inertia_path_.clear()
        self.centroids_path_.clear()

    def fit(self, X):

        self._reset_variables()
        for _ in range(self.n_init):
            centroids = self._init_centroids(X)
            for _ in range(self.max_iter):

                cluster_group = self._assign_clusters(X, centroids)
                old_centroids = centroids.copy()
                centroids = self._move_centroids(X, cluster_group)

                if np.linalg.norm(
                    centroids - old_centroids
                ) <= self.tol * np.linalg.norm(old_centroids):
                    break

            inertia = self.inertia(X, cluster_group, centroids)
            self.inertia_path_.append(np.round(inertia, 6))

            if inertia < self.inertia_:
                self.inertia_ = inertia
                self.centroids = centroids

    def _assign_clusters(self, X, centroids):
        cluster_group = []

        for row in X:
            distances = []
            for centroid in centroids:

                # distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))
                distances.append(np.dot(row - centroid, row - centroid))

            index_pos = distances.index(min(distances))
            cluster_group.append(index_pos)

        return np.array(cluster_group)

    def _move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)

    def inertia(self, X, cluster_group, centroids):
        cluster_type = np.unique(cluster_group)
        model_inertia = 0

        for type, centroid in zip(cluster_type, centroids):
            for row in X[cluster_group == type]:
                model_inertia += np.dot(row - centroid, row - centroid)

        return model_inertia

    def predict(self, X):
        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X):
        self.fit(X)

        return self.predict(X)

    def _init_centroids(self, X):

        if self.init == "random":
            random_indices = np.random.choice(
                len(X), size=self.n_clusters, replace=False
            )

            return X[random_indices]

        elif self.init == "k-means++":
            return self._kmeans_plus_plus(X)

    def _kmeans_plus_plus(self, X):

        n = len(X)
        centroids = X[np.random.choice(n, size=1)]

        for _ in range(1, self.n_clusters):
            distances = []

            for row in X:
                distances.append(min(np.dot(row - c, row - c) for c in centroids))

            total_distance = sum(distances)
            probabilities = [d / total_distance for d in distances]
            new_centroid = X[np.random.choice(n, size=1, p=probabilities)]
            centroids = np.vstack([centroids, new_centroid])

        return centroids
