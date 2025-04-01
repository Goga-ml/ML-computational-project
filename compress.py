import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


#loed and preprocess the image
def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    data = np.array(img)
    original_shape = data.shape
    data = data.reshape(-1, 3)  # flatten to a 2D array, this is done so the processing is easier
    return data, original_shape


#initialize centroids
def initialize_centroids(data, k):
    random_indices = random.sample(range(data.shape[0]), k)
    centroids = data[random_indices].astype(float)
    return centroids


#assign clusters based on closest centroids
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters.append(closest_centroid)
    return np.array(clusters)


#update centroids by calculating the mean of each cluster
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = data[random.randint(0, data.shape[0] - 1)]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


#kmeans clustering
def kmeans(data, k, max_iters=10):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):  # Check for convergence
            break
        centroids = new_centroids
    return clusters, centroids


#reconstruct compressed image
def compress_image(data, clusters, centroids, original_shape):
    compressed_data = centroids[clusters].astype(np.uint8)
    compressed_image = compressed_data.reshape(original_shape)
    return compressed_image


#run the compression algorithm
def compress_image_with_kmeans(image_path, k, max_iters=10):
    data, original_shape = load_image(image_path)
    clusters, centroids = kmeans(data, k, max_iters)
    compressed_image = compress_image(data, clusters, centroids, original_shape)

    os.makedirs('Image', exist_ok=True)
    compressed_image_path = os.path.join('Image', f"compressed_k{k}.jpg")
    Image.fromarray(compressed_image).save(compressed_image_path)


k = 16 #set the number of colors that you want the compressed image to have, the lower, the smaller the image will be but also uglier.
image_path = 'image/sky.jpg' #specify the path to the image that you want to compress
save_path = f'Image/compressed_k{k}.jpg' #specify the path where you want to save the compressed image
#compress_image_with_kmeans(image_path, k)

print("Original: "+str(os.path.getsize(image_path))+" KB") #prints the size of the original image
print("Compressed: "+str(os.path.getsize(save_path))+" KB")#prints the size of the compressed image