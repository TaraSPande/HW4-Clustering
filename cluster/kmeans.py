import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:                        #Error handle edge cases
            raise ValueError("k must be positive and an integer")

        if tol < 0:
            raise ValueError("tol must be positive or 0")

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be positive and an integer")

        self.k = k                                                  #initalize class parameters
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = None
        self.error = None
        self.n_features = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        n_obs, n_features = mat.shape
        assert self.k <= n_obs                      #Handle edge case error

        self.n_features = n_features
        rng = np.random.default_rng()               #random number generator
        indices = rng.choice(n_obs, size=self.k, replace=False)     #randomly initialize centroids
        self.centroids = mat[indices]

        prev_error = np.inf

        for _ in range(self.max_iter):              #Lloyd's Algorithm (main loop)
            distances = cdist(mat, self.centroids, metric="euclidean")      #assign distance and labels
            labels = np.argmin(distances, axis=1)                           #cdist works in hgih dimensions :)

            new_centroids = np.zeros_like(self.centroids)                   #recompute centroids and update
            for i in range(self.k):                                         #loop over each cluster
                cluster_points = mat[labels == i]
                if len(cluster_points) == 0:                                #empty cluster handling
                    new_centroids[i] = mat[rng.integers(0, n_obs)]
                else:
                    new_centroids[i] = cluster_points.mean(axis=0)          #else: compute mean

            error = np.mean(                                                #compute new error (MSE)
                np.min(cdist(mat, new_centroids, metric="euclidean") ** 2, axis=1)
            )

            if abs(prev_error - error) < self.tol:                          #check convergence
                break

            self.centroids = new_centroids
            prev_error = error

        self.error = error

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        distances = cdist(mat, self.centroids, metric="euclidean")      #compute distance to learned centroids
        return np.argmin(distances, axis=1)                             #assign each point to nearest centroid

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.error is None:
            raise RuntimeError("Model has not been fit yet")
        return self.error                   #just return error we already calcuated in fit

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise RuntimeError("Model has not been fit yet")
        return self.centroids               #just return centroids already calcuated in fit
