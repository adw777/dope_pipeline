import numpy as np
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import List, Tuple, Dict, Optional

# Initialize HDBSCAN with specified parameters
clusterer = HDBSCAN(
    min_cluster_size=2,
    allow_single_cluster=True,
    cluster_selection_method='leaf',
    store_centers='medoid'
)


def evaluateClustering(embeddings: List[List[float]], labels: List[int]) -> Dict[str, Optional[float]]:
    """
    Evaluates clustering using different metrics.

    Args:
        embeddings (List[List[float]]): The embeddings.
        labels (List[int]): The cluster labels.

    Returns:
        Dict[str, Optional[float]]: The evaluation metrics.
    """
    try:
        if len(set(labels)) > 1:  # Ensure there is more than one cluster
            silhouette = silhouette_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        else:
            silhouette = davies_bouldin = calinski_harabasz = None
        
        return {
            "silhouette_score": silhouette,
            "davies_bouldin_score": davies_bouldin,
            "calinski_harabasz_score": calinski_harabasz
        }
    except Exception as e:
        print(f"Error during clustering evaluation: {e}")
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None
        }

def clusterEmbeddings(embeddings: List[List[float]]) -> List[int]:
    """
    Clusters the embeddings and evaluates the clusters.

    Args:
        embeddings (List[List[float]]): The embeddings to be clustered.
        reduce_dimensionality (bool): Whether to reduce dimensionality before clustering.
        reduction_method (str): The method to use for dimensionality reduction ('umap' or 'pca').
    Returns:
        List[int]: The cluster labels.
    """
    try:
        if len(embeddings) == 0 or not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings array is empty or not a valid numpy array.")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got {embeddings.ndim}D array instead.")
    
        clusters = clusterer.fit_predict(embeddings)
        return clusters
    except Exception as e:
        print(f"Error during clustering: {e}")
        return []

def findMostSimilarChunks(chunks: List[str], clusters: np.ndarray, embeddings: np.ndarray) -> Tuple[List[str], List[str]]:
    """
    Finds the most similar chunk to the medoid of each cluster and returns the chunks.
    Includes noise points (outliers) in a separate list.

    Args:
        chunks (List[str]): The original text chunks.
        clusters (np.ndarray): The cluster assignments for each chunk.
        embeddings (np.ndarray): The embeddings of the chunks.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - the most representative chunks for each cluster
                                     and the outliers (noise points).
    """
    mostSimilarChunks = []
    outliers = []

    for clusterId in np.unique(clusters):
        if clusterId == -1: 
            outliers.extend([f"OUTLIER: {chunks[i]}" for i in np.where(clusters == -1)[0]])
        else:
            medoidIndex = clusterer.medoids_[clusterId]
            medoidIndex = int(medoidIndex[0]) if isinstance(medoidIndex, np.ndarray) else int(medoidIndex) 
            clusterIndices = np.where(clusters == clusterId)[0]
            distances = np.linalg.norm(embeddings[clusterIndices] - embeddings[medoidIndex], axis=1)
            representativeChunk = chunks[clusterIndices[np.argmin(distances)]]
            mostSimilarChunks.append(representativeChunk)

    return mostSimilarChunks, outliers

def clusteringPipeline(chunks: List[str], embeddings: np.ndarray) -> Tuple[List[str], List[str]]:
    """
    Full clustering pipeline that clusters embeddings and finds the most similar chunks using medoids.
    Includes noise points (outliers) in the result.

    Args:
        chunks (List[str]): The original text chunks.
        embeddings (np.ndarray): The embeddings of the chunks.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - the most representative chunks for each cluster
                                     and the outliers (noise points).
    """
    clusters = clusterEmbeddings(embeddings)
    mostSimilarChunks, outliers = findMostSimilarChunks(chunks, clusters, embeddings)
    evaluation_metrics = evaluateClustering(embeddings, clusters)
    # return mostSimilarChunks, outliers, clusters, evaluation_metrics
    return mostSimilarChunks, outliers