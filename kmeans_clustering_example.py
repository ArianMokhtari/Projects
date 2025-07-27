https://github.com/ArianMokhtari/Projects/new/main
"""
K-means Clustering Example
This script demonstrates how to perform K-means clustering using scikit-learn on the Iris dataset.

Steps:
1. Load the Iris dataset.
2. Standardize the features.
3. Fit a KMeans model with k clusters.
4. Print cluster centers and labels.
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Iris dataset
data = load_iris()
X = data.data

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit KMeans model
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Print results
print("Cluster centers (standardized space):")
print(kmeans.cluster_centers_)
print("\nCluster labels for the first 10 samples:")
print(kmeans.labels_[:10])
