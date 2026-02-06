# Write your k-means unit tests here
import pytest
import cluster
from sklearn.cluster import KMeans

def test_kmeans_0k():                                       #k=0 should return ValueError
    with pytest.raises(ValueError):
        kmeans = cluster.KMeans(0)

def test_kmeans_float_k():                                  #k type float should return ValueError
    with pytest.raises(ValueError):
        kmeans = cluster.KMeans(1.4)

def test_kmeans_neg_tol():                                  #negative tolerance should return ValueError
    with pytest.raises(ValueError):
        kmeans = cluster.KMeans(10, tol = -5)

def test_kmeans_0max_iter():                                #max_iter=0 should return ValueError
    clusters, labels = cluster.make_clusters(n=100, k=10)
    with pytest.raises(ValueError):
        kmeans = cluster.KMeans(10, max_iter = 0)

def test_kmeans_neg_max_iter():                             #negative max_iter should return ValueError
    with pytest.raises(ValueError):
        kmeans = cluster.KMeans(10, max_iter = -5)

def test_kmeans_float_max_iter():                           #max_iter type float should return ValueError
    with pytest.raises(ValueError):
        kmeans = cluster.KMeans(10, max_iter = 1.4)

def test_kmeans_obs_feats():                                #observations needs to be >= number of clusters
    clusters, labels = cluster.make_clusters(n=10, k=10)
    kmeans = cluster.KMeans(100)
    with pytest.raises(AssertionError):
        kmeans.fit(clusters)

def test_kmeans_no_fit_error():                             #try getting MSE without fitting model
    clusters, labels = cluster.make_clusters(n=100, k=10)
    kmeans = cluster.KMeans(10)
    with pytest.raises(RuntimeError):
        kmeans.get_error()

def test_kmeans_no_centroids():                             #try getting centroids without fitting model
    clusters, labels = cluster.make_clusters(n=100, k=10)
    kmeans = cluster.KMeans(10)
    with pytest.raises(RuntimeError):
        kmeans.get_centroids()

def test_kmeans_loose_mse():                                #run kmeans on loose clustering; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=10, scale=2)
    my_kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    my_kmeans.fit(clusters)

    sklearn_kmeans = KMeans(n_clusters=10, tol=1e-6, max_iter=100)
    sklearn_kmeans.fit(clusters)

    my_error = my_kmeans.get_error()
    sklearn_error = sklearn_kmeans.inertia_ / 1000          #SSE / N = MSE

    assert abs(my_error - sklearn_error) < 2

def test_kmeans_tight_mse():                                #run kmeans on tight clustering; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=10, scale=0.3)
    my_kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    my_kmeans.fit(clusters)

    sklearn_kmeans = KMeans(n_clusters=10, tol=1e-6, max_iter=100)
    sklearn_kmeans.fit(clusters)

    my_error = my_kmeans.get_error()
    sklearn_error = sklearn_kmeans.inertia_ / 1000          #SSE / N = MSE

    assert abs(my_error - sklearn_error) < 5

def test_kmeans_high_clusters():                            #run kmeans on high cluster count; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=50)
    my_kmeans = cluster.KMeans(50, tol=1e-6, max_iter=100)
    my_kmeans.fit(clusters)

    sklearn_kmeans = KMeans(n_clusters=50, tol=1e-6, max_iter=100)
    sklearn_kmeans.fit(clusters)

    my_error = my_kmeans.get_error()
    sklearn_error = sklearn_kmeans.inertia_ / 1000          #SSE / N = MSE

    assert abs(my_error - sklearn_error) < 0.25

def test_kmeans_one_dim():                                  #run kmeans on one feature; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=10, m=1)
    my_kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    my_kmeans.fit(clusters)

    sklearn_kmeans = KMeans(n_clusters=10, tol=1e-6, max_iter=100)
    sklearn_kmeans.fit(clusters)

    my_error = my_kmeans.get_error()
    sklearn_error = sklearn_kmeans.inertia_ / 1000          #SSE / N = MSE

    assert abs(my_error - sklearn_error) < 0.25

def test_kmeans_high_dim_mse():                             #run kmeans on large features; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=10, m=50)
    my_kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    my_kmeans.fit(clusters)

    sklearn_kmeans = KMeans(n_clusters=10, tol=1e-6, max_iter=100)
    sklearn_kmeans.fit(clusters)

    my_error = my_kmeans.get_error()
    sklearn_error = sklearn_kmeans.inertia_ / 1000          #SSE / N = MSE

    assert abs(my_error - sklearn_error) < 500


#MY TESTING VISUALLY BY PLOTTING THE CLUSTERS :)

# kmeans = cluster.KMeans(10)
# clusters, labels = cluster.make_clusters(n=1000, k=10, scale=0.3)
# kmeans.fit(clusters)
# pred_labels = kmeans.predict(clusters)

# print(kmeans.get_error())
# print(kmeans.get_centroids())

# cluster.plot_clusters(clusters, labels)
# cluster.plot_clusters(clusters, pred_labels)
