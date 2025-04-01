import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import k_means_psychographic

df = pd.read_csv("data/behavior.csv") #specify you data path here
#I drop useless, non-numeric datatypes that does not contribute much
df = df.select_dtypes(include=[np.number])
df.drop(['Customer ID'], axis=1, inplace=True)
print(df.dtypes)


# elbow point was 4, so I use 4 clusters
k = 4
kmeans = k_means_psychographic
kmeans.initialize_centroids(df, k)
clusters, centroids = kmeans.kmeans(df, k)

# Use the x and y coordinates from the clustered data
x_feature = df.iloc[:, 0]  # Total Spend
y_feature = df.iloc[:, 1]  # Items Purchased


# Plot the clustered data
plt.figure(figsize=(10, 7))
for i in range(k):
    cluster_points = df[clusters == i]
    plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Cluster {i}', s=50, alpha=0.6)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')

# Chart aesthetics
plt.xlabel('Total Spend (Normalized)')
plt.ylabel('Items Purchased (Normalized)')
plt.title('K-Means Clustering (k=4) on Total Spend vs Items Purchased')
plt.legend()
plt.grid(True)
plt.show()
