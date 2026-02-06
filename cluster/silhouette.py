import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        n_samples = X.shape[0]
        labels = np.unique(y)

        D = cdist(X, X, metric="euclidean")             #compare all pairwise distances

        silhouettes = np.zeros(n_samples)

        for i in range(n_samples):                      #compute silhouette for each point
            same_cluster = y == y[i]                    #find points in same cluster (boolean mask)
            other_clusters = labels[labels != y[i]]     #...other clusters

            if np.sum(same_cluster) > 1:                #intra-cluster distance
                a_i = np.mean(D[i, same_cluster & (np.arange(n_samples) != i)])
            else:
                a_i = 0.0                               #singleton cluster

            b_i = np.inf
            for label in other_clusters:                #loop over other clusters
                mask = y == label
                b_i = min(b_i, np.mean(D[i, mask]))

            if max(a_i, b_i) == 0:                      #compute silhouette formula
                silhouettes[i] = 0.0
            else:
                silhouettes[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouettes
