"""
DESCRIPTION: unsupervised learning with DBSCAN.
AUTHORS: ...
DATE: 8/11/21
"""

# MODULES IMPORT
from sklearn.cluster import DBSCAN
# TODO Import additional required modules here

# DBSCAN HYPERPARAMETERS DEFINITION
minimum_points = None  # TODO Set this hyperparameter properly
epsilon = None  # TODO Set this hyperparameter properly

# DBSCAN ESTIMATOR INSTANTIATION
dbscan = DBSCAN(eps=epsilon, min_samples=minimum_points)

# DBSCAN CLUSTERING
# TODO Perform DBSCAN clustering

# CLUSTERS ANALYSIS
# Clusters extraction
# TODO Extract clusters and outlier points
number_clusters = None  # TODO Calculate the number of clusters

# Clusters exploration
# TODO Explore your clusters and noisy points using a scatter plot with colors identifying each cluster/outlier point.

# POINT TYPES ANALYSIS
# TODO Generate a vector array (or a list) to indicate if the point is a core point (identified by 1),
# TODO a border point (identified by 0) or an outlier point (identified by -1). For example, your point_types object
# TODO could be [1, 1, 1, 0, 1, 1, -1, 1, 0, ...]
point_types = None  # TODO Get the point types

# Point types exploration
# TODO Explore your core and noisy points using a scatter plot with colors identifying each cluster/outlier point.