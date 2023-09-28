import os
import numpy as np
from sklearn.metrics import (
    pairwise_distances,
    silhouette_score,
    calinski_harabasz_score)
from sklearn.cluster import MiniBatchKMeans


# %%

def kmeans_clustering(
        data: np.ndarray,
        n_clusters: int,
        return_centroids: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Perform K-means clustering on the given data and return
    the cluster labels and centroids."""
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=1234,
        n_init='auto',
        batch_size=256 * os.cpu_count() + 1
    )
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    if return_centroids:
        return labels, centroids
    return labels


def calculate_dunn_index(
        data: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray) -> float:
    """Calculate the Dunn Index for evaluating cluster quality."""
    cluster_distances = []
    for cluster_label in np.unique(labels):
        cluster_points = data[labels == cluster_label]
        if len(cluster_points) > 1:
            intra_cluster_distances = pairwise_distances(
                cluster_points, metric='euclidean', n_jobs=-1
            )
            cluster_distances.append(np.mean(intra_cluster_distances))

    inter_cluster_distances = pairwise_distances(
        centroids, metric='euclidean', n_jobs=-1
    )
    min_inter_cluster_distance = np.min(
        inter_cluster_distances[inter_cluster_distances > 0]
    )
    max_intra_cluster_distance = np.max(cluster_distances)
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index


def find_optimal_clusters(
        data: np.ndarray,
        min_clusters: int = 6,
        max_clusters: int = 8) -> int:
    """Find the optimal number of clusters using
    three metrics simultaneously."""
    metrics = {}

    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            labels, centroids = kmeans_clustering(data, n_clusters)
            score_silhouette = silhouette_score(data, labels)
            score_calinski = calinski_harabasz_score(data, labels)
            score_dunn = calculate_dunn_index(data, labels, centroids)
            metrics[n_clusters] = (
                score_silhouette, score_calinski, score_dunn
            )

            print(score_silhouette, score_calinski, score_dunn)

        except ValueError:
            continue  # Skip to the next iteration if an error occurs

    # Select the optimal number of clusters
    sorted_clusters = sorted(
        metrics.keys(), key=lambda x: metrics[x], reverse=True
    )
    optimal_clusters = sorted_clusters[0]

    return optimal_clusters
