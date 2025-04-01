import random
import numpy as np


def initialize_centroids(data, k):
    # Randomly selecting k data points as initial centroids
    data = data.select_dtypes(include=[np.number]).values
    random_indices = random.sample(range(data.shape[0]), k)
    centroids = data[random_indices].astype(float)
    return centroids


def assign_clusters(data, centroids):
    # Assign each data point to the nearest centroid
    clusters = []
    data = data.select_dtypes(include=[np.number]).values

    for point in data:
        distances = [np.linalg.norm(point - centroid, ord=2) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters.append(closest_centroid)
    return np.array(clusters)


def update_centroids(data, clusters, k):
    # Update centroids by calculating the mean of each cluster
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = data[random.randint(0, data.shape[0] - 1)]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


def kmeans(data, k, max_iters=400):
    centroids = initialize_centroids(data, k)
    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids


