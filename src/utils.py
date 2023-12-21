import ot
import numpy as np


def optimal_transport_matching(clusters, y, n_clusters=None):
    if n_clusters is None:
        n_clusters = max(len(np.unique(clusters)), len(np.unique(y)))

    pred_clusters = clusters

    clusters_histograms = np.zeros((n_clusters, n_clusters))
    for pred, true in zip(pred_clusters, y):
        clusters_histograms[int(pred) - 1, int(true) - 1] += 1

    a = np.ones(n_clusters)
    b = np.ones(n_clusters)

    M = clusters_histograms.max(axis=0).reshape(1, -1) - clusters_histograms
    G = ot.emd(a, b, M)

    col_max = np.argmax(G, axis=1)

    pred_labels = {i: col_max[i] for i in range(n_clusters)}
    return pred_labels


def optimal_transport_matching_labels(clusters, y, target_decoder, n_clusters):
    pred_clusters = clusters

    clusters_histograms = np.zeros((n_clusters, n_clusters))
    for pred, true in zip(pred_clusters, y):
        clusters_histograms[int(pred) - 1, int(true) - 1] += 1

    a = np.ones(n_clusters)
    b = np.ones(n_clusters)

    M = clusters_histograms.max(axis=0).reshape(1, -1) - clusters_histograms
    G = ot.emd(a, b, M)

    col_max = np.argmax(G, axis=1)

    pred_labels = {i + 1: target_decoder[col_max[i] + 1] for i in range(n_clusters)}
    return pred_labels
