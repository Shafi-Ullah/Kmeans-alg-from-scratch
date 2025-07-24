import numpy as np


class KMeans:
    def __init__(
        self, n_clusters=8, max_iter=300, init="random", n_init=10, tol=0.0001
    ):
        """
        Custom implementation of the K-Means clustering algorithm.

        Args:
            n_clusters (int): Number of clusters to form.
            max_iter (int): Maximum number of iterations per initialization.
            init (str): Initialization method - 'random' or 'k-means++'.
            n_init (int): Number of times the algorithm will be run with different centroid seeds.
            tol (float): Relative tolerance to declare convergence.
        """

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
        """Reset all internal variables before a fresh `fit()`."""
        self.centroids = None
        self.inertia_ = np.inf
        self.inertia_path_.clear()
        self.centroids_path_.clear()

    def fit(self, X: np.array):
        """
        Compute k-means clustering.

        Args:
            X (np.ndarray): Training instances to cluster of shape (n_samples, n_features).
        """
        self._reset_variables()
        for _ in range(self.n_init):
            centroids = self._init_centroids(X)
            for _ in range(self.max_iter):

                cluster_group = self._assign_clusters(X, centroids)
                old_centroids = centroids.copy()
                centroids = self._move_centroids(X, cluster_group, old_centroids)

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
        """
        Assign each data point in X to the nearest centroid.

        Args:
            X (np.ndarray): Data points of shape (n_samples, n_features).
            centroids (np.ndarray): Current centroids of shape (n_clusters, n_features).

        Returns:
            np.ndarray: Array of cluster indices for each sample.
        """
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        cluster_group = np.argmin(distances, axis=1)

        return cluster_group

    def _move_centroids(self, X, cluster_group, old_centroids):
        """
        Update centroids as the mean of assigned data points.

        Args:
            X (np.ndarray): Data points of shape (n_samples, n_features).
            cluster_group (np.ndarray): Cluster index for each data point.
            old_centroids (np.ndarray): Centroids before the update.

        Returns:
            np.ndarray: Updated centroids.
        """
        new_centroids = []
        # Loop over all clusters, not just the ones that received(old_centroids) points
        for type in range(self.n_clusters):
            points = X[cluster_group == type]
            # Fallback: retain old centroid
            # If a cluster has no points (an empty cluster), then the centroid of that cluster is kept at its previous position. So that the number of centroids(return value of this function) remains equal to the self.n_clusters.

            if len(points) == 0:
                new_centroids.append(old_centroids[type])
            # Update centroid using mean
            else:
                new_centroids.append(points.mean(axis=0))

        return np.array(new_centroids)

    def inertia(self, X, cluster_group, centroids):
        """
        Compute the inertia (sum of squared distances to closest centroid).

        Args:
            X (np.ndarray): Data points.
            cluster_group (np.ndarray): Assigned cluster indices.
            centroids (np.ndarray): Final centroids.

        Returns:
            float: Total inertia.
        """

        # differences: for each sample, subtract its assigned centroid
        differences = X - centroids[cluster_group]

        # squared_distances: compute squared distance from each sample to its centroid
        squared_distances = np.sum(differences**2, axis=1)

        # model_inertia: sum of all these distances
        model_inertia = squared_distances.sum()

        return model_inertia

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Args:
            X (np.ndarray): Data to predict.

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """

        self.fit(X)

        return self.predict(X)

    def _init_centroids(self, X):
        """
        Initialize centroids using specified strategy --> random, k-means++.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: Initialized centroids.
        """

        if self.init == "random":
            random_indices = np.random.choice(
                len(X), size=self.n_clusters, replace=False
            )

            return X[random_indices]

        elif self.init == "k-means++":
            return self._kmeans_plus_plus(X)

    def _kmeans_plus_plus(self, X):
        """
        Initialize centroids using the K-Means++ algorithm.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: Centroids initialized using K-Means++.
        """

        n = len(X)
        centroids = X[np.random.choice(n, size=1)]

        for _ in range(1, self.n_clusters):

            distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

            min_distance_index = np.argmin(distances, axis=1)

            minimum_distance = distances[np.arange(len(distances)), min_distance_index]

            total_distance = sum(minimum_distance)
            probabilities = [d / total_distance for d in minimum_distance]
            new_centroid = X[np.random.choice(n, size=1, p=probabilities)]
            centroids = np.vstack([centroids, new_centroid])

        return centroids
