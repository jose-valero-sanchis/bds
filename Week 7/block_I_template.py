"""
DESCRIPTION: unsupervised learning with DBSCAN.
AUTHORS: Jose Valero & Lucas Fayolle
DATE: 21/10/24
"""

# MODULES IMPORT
from scipy.spatial.distance import pdist
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

# DATA LOADING
df = pd.read_csv("data.csv")

# DATA EXPLORATION
plt.figure(figsize=(8,6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.title('Scatter Plot')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.grid(True)
# plt.show()

# DISTANCES VECTOR
points = df.values
distances = np.sqrt(np.sum((points[:, np.newaxis] - points[np.newaxis, :]) ** 2, axis=-1))
dist_vector = distances[np.triu_indices(len(points), k=1)] 

# DISTANCES VECTOR EXPLORATION
# Histogram
# TODO Draw an histogram to represent the distances, adjusting the number of bins
# Histogram of distances
plt.figure(figsize=(8,6))
plt.hist(dist_vector, bins=30, color='skyblue', edgecolor='black')  # You can adjust 'bins' as needed
plt.title('Histogram of Pairwise Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Vector sorting
# TODO Sort your distances in ascending order
sorted_distances = np.sort(dist_vector)

# Line plot
plt.figure(figsize=(8,6))
plt.plot(sorted_distances, marker='o', linestyle='-', color='b')
plt.title('Sorted Euclidean Distances')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# K-GRAPH MATRIX
# TODO Get the K-graph distance matrix

"""
Automatizar para obtener, para diferentes valores de k, su valor correspondiente de epsilon.
Calcular incremento en y para decidir donde cortar, porque x están equiespaciadas.
"""

k = 4

# 2. Usar NearestNeighbors para calcular las distancias de los k-vecinos más cercanos
nbrs = NearestNeighbors(n_neighbors=k).fit(points)
distances, indices = nbrs.kneighbors(points)

# 3. Tomar la distancia al k-ésimo vecino más cercano para cada punto
k_distances = distances[:, k-1]  # El último vecino es el k-ésimo

# 4. Ordenar las distancias
k_distances_sorted = np.sort(k_distances)

# 5. Graficar el k-distance graph
plt.figure(figsize=(8,6))
plt.plot(k_distances_sorted)
plt.title(f'{k}-distance graph')
plt.xlabel('Points sorted according to distance')
plt.ylabel(f'{k}-nearest neighbor distance')
plt.grid(True)
plt.show()