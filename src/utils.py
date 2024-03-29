"""
 Copyright (c) 2024 Théo Rudkiewicz, Thomas Michel, Ali Ramlaoui

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""
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


def optimal_transport_matching_labels(clusters, y, target_decoder):
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

    pred_labels = {i + 1: target_decoder[col_max[i] + 1] for i in range(n_clusters)}
    return pred_labels
