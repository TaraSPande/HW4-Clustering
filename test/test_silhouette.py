# write your silhouette score unit tests here
import pytest
import cluster
import numpy as np
from sklearn.metrics import silhouette_score


def test_kmeans_loose_mse():                    #run kmeans on loose clustering; compare silhouette scores
    clusters, labels = cluster.make_clusters(n=1000, k=10, scale=2)
    kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    kmeans.fit(clusters)
    pred_labels = kmeans.predict(clusters)

    my_silhouette = cluster.Silhouette()
    my_scores = my_silhouette.score(clusters, pred_labels)
    my_mean_score = np.mean(my_scores)

    sklearn_score = silhouette_score(clusters, pred_labels)

    assert np.round(my_mean_score, 5) == np.round(sklearn_score, 5)     #accurate to 5 decimals

def test_kmeans_tight_mse():                    #run kmeans on tight clustering; compare silhouette scores
    clusters, labels = cluster.make_clusters(n=1000, k=10, scale=0.3)
    kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    kmeans.fit(clusters)
    pred_labels = kmeans.predict(clusters)

    my_silhouette = cluster.Silhouette()
    my_scores = my_silhouette.score(clusters, pred_labels)
    my_mean_score = np.mean(my_scores)

    sklearn_score = silhouette_score(clusters, pred_labels)

    assert np.round(my_mean_score, 5) == np.round(sklearn_score, 5)     #accurate to 5 decimals

def test_kmeans_high_clusters():                            #run kmeans on high cluster count; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=50)
    kmeans = cluster.KMeans(50, tol=1e-6, max_iter=100)
    kmeans.fit(clusters)
    pred_labels = kmeans.predict(clusters)

    my_silhouette = cluster.Silhouette()
    my_scores = my_silhouette.score(clusters, pred_labels)
    my_mean_score = np.mean(my_scores)

    sklearn_score = silhouette_score(clusters, pred_labels)

    assert np.round(my_mean_score, 5) == np.round(sklearn_score, 5)     #accurate to 5 decimals

def test_kmeans_one_dim():                                  #run kmeans on one feature; compare mse
    clusters, labels = cluster.make_clusters(n=1000, k=10, m=1)
    kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    kmeans.fit(clusters)
    pred_labels = kmeans.predict(clusters)

    my_silhouette = cluster.Silhouette()
    my_scores = my_silhouette.score(clusters, pred_labels)
    my_mean_score = np.mean(my_scores)

    sklearn_score = silhouette_score(clusters, pred_labels)

    assert np.round(my_mean_score, 5) == np.round(sklearn_score, 5)     #accurate to 5 decimals

def test_kmeans_high_dim_mse():                 #run kmeans on large features; compare silhouette scores
    clusters, labels = cluster.make_clusters(n=1000, k=10, m=50)
    kmeans = cluster.KMeans(10, tol=1e-6, max_iter=100)
    kmeans.fit(clusters)
    pred_labels = kmeans.predict(clusters)

    my_silhouette = cluster.Silhouette()
    my_scores = my_silhouette.score(clusters, pred_labels)
    my_mean_score = np.mean(my_scores)

    sklearn_score = silhouette_score(clusters, pred_labels)

    assert np.round(my_mean_score, 5) == np.round(sklearn_score, 5)     #accurate to 5 decimals
