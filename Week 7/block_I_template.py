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
plt.figure(figsize=(8,6))
plt.hist(dist_vector, bins=30, color='skyblue', edgecolor='black')
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

D = df.shape[1]  
tau_min = D + 1  
tau_max = 10   

results = []

for tau in range(tau_min, tau_max + 1):
    k = tau 

    nbrs = NearestNeighbors(n_neighbors=k).fit(df)
    distances, indices = nbrs.kneighbors(df)

    k_distances = distances[:, k - 1]
    k_distances_sorted = np.sort(k_distances)

    dy = np.diff(k_distances_sorted)
    dx = np.ones_like(dy)

    angles = np.degrees(np.arctan(dy / dx))

    significant_angle_indices = np.where(angles > 1.5)[0]

    threshold_index = int(len(dy) * 0.1)
    significant_angle_indices = significant_angle_indices[significant_angle_indices > threshold_index]

    if significant_angle_indices.size > 0:
        significant_angle_index = significant_angle_indices[0]
        epsilon = k_distances_sorted[significant_angle_index]
    else:
        significant_angle_index = len(k_distances_sorted) - 1
        epsilon = k_distances_sorted[-1]

    results.append({
        'tau': tau,
        'k': k,
        'epsilon': epsilon,
        'significant_index': significant_angle_index
    })

    plt.figure(figsize=(8, 6))
    plt.plot(k_distances_sorted, marker='o', linestyle='-', color='b', label=f'k={k}')
    plt.axvline(x=significant_angle_index, color='r', linestyle='--', label=f"Corte en ε={epsilon:.3f}")
    plt.title(f'Gráfico de Distancia Ordenada para k={k} (tau={tau})')
    plt.xlabel('Puntos ordenados por distancia')
    plt.ylabel(f'{k}-ésimo vecino más cercano')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Para tau (MinPts) = {tau}, k = {k}, el valor estimado de epsilon es {epsilon:.3f}")