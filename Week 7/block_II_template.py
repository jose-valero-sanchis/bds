"""
DESCRIPTION: unsupervised learning with DBSCAN.
AUTHORS: Jose Valero & Lucas Fayolle
DATE: 21/10/24
"""

# MODULES IMPORT
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")

params = [
    {'tau': 3, 'epsilon': 1.903},
    {'tau': 4, 'epsilon': 2.357},
    {'tau': 5, 'epsilon': 2.679},
    {'tau': 6, 'epsilon': 2.733},
    {'tau': 7, 'epsilon': 2.813},
    {'tau': 8, 'epsilon': 3.161},
    {'tau': 9, 'epsilon': 3.395},
    {'tau': 10, 'epsilon': 3.721}
]

results = []

for param in params:
    min_samples = int(param['tau'])
    epsilon = param['epsilon']
    
    # DBSCAN ESTIMATOR INSTANTIATION
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    
    # DBSCAN CLUSTERING
    labels = dbscan.fit_predict(df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Compute silhouette score if possible
    if n_clusters > 1 and n_clusters < len(df):
        try:
            score = silhouette_score(df, labels)
            results.append({
                'min_samples': min_samples,
                'epsilon': epsilon,
                'n_clusters': n_clusters,
                'silhouette_score': score
            })
        except:
            continue
    else:
        continue

results_df = pd.DataFrame(results)

if not results_df.empty:
    best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
    best_min_samples = int(best_result['min_samples'])  
    best_epsilon = best_result['epsilon']

    print(f"Best MinPts: {best_min_samples}, Best Epsilon: {best_epsilon:.3f}")
    print(f"Number of clusters: {int(best_result['n_clusters'])}, Silhouette Score: {best_result['silhouette_score']:.4f}")

    # DBSCAN ESTIMATOR INSTANTIATION WITH BEST HYPERPARAMETERS
    dbscan = DBSCAN(eps=best_epsilon, min_samples=best_min_samples)

    # DBSCAN CLUSTERING
    dbscan.fit(df)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # CLUSTERS ANALYSIS
    # Clusters exploration
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8, 6))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
        xy = df.values[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title(f'DBSCAN Clustering (Estimated number of clusters: {n_clusters})')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.show()
    
    # POINT TYPES ANALYSIS
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    
    plt.figure(figsize=(8,6))
    
    # Core points
    core_points = df.values[(labels != -1) & core_samples_mask]
    plt.plot(core_points[:,0], core_points[:,1], 'o', markerfacecolor='b',
             markeredgecolor='k', markersize=6, label='Core points')
    
    # Border points
    border_points = df.values[(labels != -1) & ~core_samples_mask]
    plt.plot(border_points[:,0], border_points[:,1], 'o', markerfacecolor='c',
             markeredgecolor='k', markersize=6, label='Border points')
    
    # Noise points
    noise_points = df.values[labels == -1]
    plt.plot(noise_points[:,0], noise_points[:,1], 'o', markerfacecolor='r',
             markeredgecolor='k', markersize=6, label='Noise points')
    
    plt.legend()
    plt.title('DBSCAN Point Types')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.show()
else:
    print("No valid clustering results were found with the provided parameters.")