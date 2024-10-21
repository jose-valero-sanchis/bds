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

# DISTANCES VECTOR EXPLORATION
# Histogram
# TODO Draw an histogram to represent the distances, adjusting the number of bins

# Vector sorting
# TODO Sort your distances in ascending order

# Line plot
# TODO Draw a line plot to represent the sorted distances

# K-GRAPH MATRIX
# TODO Get the K-graph distance matrix


