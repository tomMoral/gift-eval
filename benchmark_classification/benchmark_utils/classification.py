"""Classification utilities for embedding-based models."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit norm.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_samples, embedding_dim)
    
    Returns
    -------
    np.ndarray
        Normalized embeddings
    """
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)


def compute_class_centroids(embeddings, labels, num_classes):
    """Compute class centroids from embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_samples, embedding_dim)
    labels : np.ndarray
        Array of shape (n_samples,) with class labels
    num_classes : int
        Number of classes
    
    Returns
    -------
    np.ndarray
        Class centroids of shape (num_classes, embedding_dim)
    """
    centroids = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
        else:
            centroid = np.zeros(embeddings.shape[1])
        centroids.append(centroid)
    
    return np.array(centroids)


def predict_nearest_centroid(embeddings, centroids):
    """Predict class using nearest centroid classifier.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_test_samples, embedding_dim)
    centroids : np.ndarray
        Array of shape (num_classes, embedding_dim)
    
    Returns
    -------
    y_pred : np.ndarray
        Predicted class labels (n_test_samples,)
    distances : np.ndarray
        Distances to each class (n_test_samples, num_classes)
    """
    distances = np.linalg.norm(
        embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2,
    )
    y_pred = np.argmin(distances, axis=1)
    return y_pred, distances


def distances_to_probabilities(distances, temperature=1.0):
    """Convert distances to class probabilities using softmax.
    
    Parameters
    ----------
    distances : np.ndarray
        Array of shape (n_samples, num_classes) with distances
    temperature : float
        Temperature for softmax
    
    Returns
    -------
    np.ndarray
        Probabilities of shape (n_samples, num_classes)
    """
    # Softmax over negative distances (closer = higher probability)
    neg_distances = -distances / temperature
    exp_distances = np.exp(neg_distances - neg_distances.max(axis=1, keepdims=True))
    proba = exp_distances / exp_distances.sum(axis=1, keepdims=True)
    return proba
