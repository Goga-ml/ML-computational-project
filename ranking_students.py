import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#Generate synthetic data (each student has a 4x5 matrix with values from 51 to 100)
def generate_data(num_students=10, rows=4, cols=5, min_val=51, max_val=100):
    return [np.random.randint(min_val, max_val + 1, size=(rows, cols)) for _ in range(num_students)]


# kmeans with kmeans++ centroid initialization
def k_means_clustering(data, order, k=4, max_iterations=100):
    np.random.seed(0)
    centroids = [data[idx] for idx in np.random.choice(len(data), 1)]  #start with one random centroid
    while len(centroids) < k:
        distances = []
        for student_matrix in data:
            min_distance = min(np.linalg.norm(student_matrix - centroid, ord=order) for centroid in centroids)
            distances.append(min_distance)

        #select a new centroid with probability proportional to the distance, this is kmeans++
        prob_distribution = np.array(distances) / sum(distances)
        new_centroid_idx = np.random.choice(len(data), p=prob_distribution)
        centroids.append(data[new_centroid_idx])

    #clustering process
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for i, student_matrix in enumerate(data):
            distances = [np.linalg.norm(student_matrix - centroid, ord=order) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)

        # update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean([data[idx] for idx in cluster], axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[clusters.index(cluster)])

        # check for convergence
        if all(np.allclose(centroid, new_centroid, atol=1e-5) for centroid, new_centroid in
               zip(centroids, new_centroids)):
            break

        centroids = new_centroids

    # here wemap points to clusters for final assignment
    cluster_assignments = np.zeros(len(data))
    for cluster_idx, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_assignments[idx] = cluster_idx

    return cluster_assignments, centroids



data = generate_data(num_students=10) #set how many students you want
cluster_assignments, centroids = k_means_clustering(data, np.inf) #'fro' is frobenious norm, if you want different norm, change it


# visualization function
def visualize_clusters(data, cluster_assignments, centroids):
    flattened_data = np.array([student_matrix.flatten() for student_matrix in data])
    flattened_centroids = np.array([centroid.flatten() for centroid in centroids])

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(flattened_data)
    reduced_centroids = pca.transform(flattened_centroids)

    cluster_assignments = cluster_assignments.astype(int)

    plt.figure(figsize=(8, 6))
    unique_clusters = np.unique(cluster_assignments)
    for cluster_idx in unique_clusters:
        cluster_points = reduced_data[cluster_assignments == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {chr(65 + cluster_idx)}')

    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], color='black', marker='X', s=100, label='Centroids')

    plt.title('K-means Clustering of Student Grades')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


# visualize clusters
visualize_clusters(data, cluster_assignments, centroids)
