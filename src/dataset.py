import numpy as np
import pandas as pd
from src.utils import optimal_transport_matching_labels
from sklearn.manifold import TSNE, MDS
from src.clustering import OrdinalClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    adjusted_rand_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from collections import defaultdict
from time import time
from scipy.stats import wasserstein_distance
from src.aecm import AECM_GOD, AECM_BOS


class BaseDataset:
    def __init__(self, path, n_iter=100, eps=1e-3, silent=False, seed=0):
        """
        Base class for datasets
        :param path: path to the dataset
        :param n_iter: number of iterations for the EM algorithm
        :param eps: convergence threshold for the EM algorithm
        :param silent: if True, no output is printed
        :param seed: seed for the random number generator
        """
        self.silent = silent
        self.eps = eps
        self.path = path
        self.data = None
        self.X = None
        self.target = None
        self.target_decoder = None
        self.target_decoder_inv = None
        self.y = None
        self.clusters = None
        self.m = None
        self.seed = seed
        self.n_iter = n_iter
        self.true_labels = None
        self.last_runtype = None
        self.all_pred_labels = defaultdict(any)
        self.all_clusters = defaultdict(any)
        self.runtime = defaultdict(int)
        self.scores = defaultdict(dict)

    def compute_n_cat(self):
        """
        Compute the number of categories for each feature
        :return: None
        """
        if self.m is None:
            self.m = np.array(
                [len(self.data[col].unique()) for col in self.data.columns]
            )

    def compute_Xy(self):
        """
        Compute the input matrix X and the target vector y
        :return: None
        """
        raise NotImplementedError

    def compute_target_decoder(self):
        """
        Compute the target decoder and its inverse
        :return: None
        """
        if self.target_decoder is None:
            self.target_decoder = {
                i + 1: val for i, val in enumerate(self.data[self.target].unique())
            }
        if self.target_decoder_inv is None:
            self.target_decoder_inv = {
                val: i + 1 for i, val in enumerate(self.data[self.target].unique())
            }

    def compute_pred_labels(self):
        """
        Compute the predicted labels
        :return: None
        """
        self.pred_labels = optimal_transport_matching_labels(
            self.clusters,
            self.y,
            target_decoder=self.target_decoder,
        )

    def cluster_gaussian(self, n_clusters=None):
        """
        Cluster the data using the Gaussian model
        :param n_clusters: number of clusters
        :return: the clusters
        """
        start_time = time()
        if n_clusters is None and self.n_clusters is None:
            raise Exception("n_clusters not specified")
        if n_clusters is None:
            n_clusters = self.n_clusters
        if self.X is None or self.y is None:
            self.compute_Xy()

        self.gaussian_clustering = GaussianMixture(
            n_clusters,
            n_init=10,
            covariance_type="full",
            random_state=self.seed,
        )
        self.clusters = self.gaussian_clustering.fit_predict(self.X) + 1  # 1-indexed
        self.n_clusters_compute = n_clusters

        self.compute_target_decoder()
        self.compute_pred_labels()

        self.last_runtype = "Gaussian"
        self.all_clusters[self.last_runtype] = self.clusters
        self.all_pred_labels[self.last_runtype] = self.pred_labels

        if not self.silent:
            print("Clustered data into {} clusters".format(n_clusters))
            print(f"Estimated means: {self.gaussian_clustering.means_}")
            print(f"Estimated covariances: {self.gaussian_clustering.covariances_}")

        self.runtime[self.last_runtype] += time() - start_time
        return self.clusters

    def cluster_kmeans(self, n_clusters=None):
        """
        Cluster the data using K-Means
        :param n_clusters: number of clusters
        :return: the clusters
        """
        start_time = time()
        if n_clusters is None and self.n_clusters is None:
            raise Exception("n_clusters not specified")
        if n_clusters is None:
            n_clusters = self.n_clusters
        if self.X is None or self.y is None:
            self.compute_Xy()

        self.kmeans_clustering = KMeans(
            n_clusters,
            n_init=10,
            random_state=self.seed,
        )
        self.clusters = self.kmeans_clustering.fit_predict(self.X) + 1  # 1-indexed
        self.n_clusters_compute = n_clusters

        self.compute_target_decoder()
        self.compute_pred_labels()

        self.last_runtype = "K-Means"
        self.all_clusters[self.last_runtype] = self.clusters
        self.all_pred_labels[self.last_runtype] = self.pred_labels

        if not self.silent:
            print("Clustered data into {} clusters".format(n_clusters))
            print(f"Estimated means: {self.kmeans_clustering.cluster_centers_}")

        self.runtime[self.last_runtype] += time() - start_time
        return self.clusters

    def cluster_god(self, n_clusters=None, m=None, init="random"):
        """
        Cluster the data using the GOD algorithm
        :param n_clusters: number of clusters
        :param m: number of categories for each feature
        :return: the clusters
        """
        start_time = time()
        if n_clusters is None and self.n_clusters is None:
            raise Exception("n_clusters not specified")
        if n_clusters is None:
            n_clusters = self.n_clusters
        if m is None:
            self.compute_n_cat()
            m = self.m
        if self.X is None or self.y is None:
            self.compute_Xy()

        # self.ordinal_clustering = OrdinalClustering(
        #     n_clusters,
        #     n_iter=self.n_iter,
        #     init=init,
        #     model="god",
        #     eps=self.eps,
        #     silent=self.silent,
        #     seed=self.seed,
        # )

        self.ordinal_clustering = AECM_GOD(
            nb_clusters=n_clusters,
            nb_features=self.X.shape[1],
            ms=m,
            data=self.X,
            eps=self.eps,
            seed=self.seed,
        )

        loglikelihoods = self.ordinal_clustering.fit(
            initialization=init, epsilon_aecm=self.eps
        )
        self.clusters = self.ordinal_clustering.labels + 1

        # self.clusters = (j
        #     self.ordinal_clustering.fit_transform(self.X, m) + 1
        # )  # 1-indexed

        self.n_clusters_compute = n_clusters

        self.compute_target_decoder()
        self.compute_pred_labels()

        if init == "random":
            init_name = "Random"
        elif init == "kmeans":
            init_name = "K-Means"
        self.last_runtype = f"GOD {init_name}"
        self.all_clusters[self.last_runtype] = self.clusters
        self.all_pred_labels[self.last_runtype] = self.pred_labels

        if not self.silent:
            print("Clustered data into {} clusters".format(n_clusters))
            print(f"Estimated alpha: {self.ordinal_clustering.alphas}")
            print(f"Estimated mu: {self.ordinal_clustering.mus}")
            print(f"Estimated pi: {self.ordinal_clustering.pis}")

        self.runtime[self.last_runtype] += time() - start_time
        return self.clusters

    def cluster_bos(self, n_clusters=None, m=None, init="random"):
        """
        Cluster the data using the BOS algorithm
        :param n_clusters: number of clusters
        :param m: number of categories for each feature
        :return: the clusters
        """
        start_time = time()
        if n_clusters is None and self.n_clusters is None:
            raise Exception("n_clusters not specified")
        if n_clusters is None:
            n_clusters = self.n_clusters
        if m is None:
            self.compute_n_cat()
            m = self.m
        if self.X is None or self.y is None:
            self.compute_Xy()

        # self.ordinal_clustering = OrdinalClustering(
        #     n_clusters,
        #     n_iter=self.n_iter,
        #     model="bos",
        #     init=init,
        #     eps=self.eps,
        #     silent=self.silent,
        #     seed=self.seed,
        # )

        self.ordinal_clustering = AECM_BOS(
            nb_clusters=n_clusters,
            nb_features=self.X.shape[1],
            ms=m,
            data=self.X,
            eps=self.eps,
            seed=self.seed,
        )

        loglikelihoods = self.ordinal_clustering.fit(
            initialization=init, epsilon_aecm=self.eps
        )
        self.clusters = self.ordinal_clustering.labels + 1

        # self.clusters = (
        #     self.ordinal_clustering.fit_transform(self.X, m) + 1
        # )  # 1-indexed
        self.n_clusters_compute = n_clusters

        self.compute_target_decoder()
        self.compute_pred_labels()

        if init == "random":
            init_name = "Random"
        elif init == "kmeans":
            init_name = "K-Means"
        self.last_runtype = f"BOS {init_name}"
        self.all_clusters[self.last_runtype] = self.clusters
        self.all_pred_labels[self.last_runtype] = self.pred_labels

        if not self.silent:
            print("Clustered data into {} clusters".format(n_clusters))
            print(f"Estimated alpha: {self.ordinal_clustering.alphas}")
            print(f"Estimated mu: {self.ordinal_clustering.mus}")
            print(f"Estimated pi: {self.ordinal_clustering.pis}")

        self.runtime[self.last_runtype] += time() - start_time
        return self.clusters

    def classification_results(self, runtype=None, plot=True):
        """
        Compute the classification results
        :param plot: if True, plot the confusion matrix
        :return: the confusion matrix and the classification report
        """
        if "scores" not in dir(self):
            self.scores = defaultdict(list)

        if self.silent:
            plot = False

        if runtype is not None:
            assert runtype in self.all_clusters.keys()
            clusters = self.all_clusters[runtype]
            pred_labels = self.all_pred_labels[runtype]
            runtype = runtype
        elif self.clusters is None:
            raise Exception("Clusters not computed yet")
        else:
            runtype = self.last_runtype
            clusters = self.clusters
            pred_labels = self.pred_labels

        y_pred = np.zeros_like(self.y)
        for i in range(len(self.y)):
            y_pred[i] = self.target_decoder_inv[pred_labels[clusters[i]]]

        cm = confusion_matrix(self.y, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.target_decoder.values()
        )
        if plot:
            disp.plot()
            plt.title(f"Confusion matrix ({runtype})")

        cr = classification_report(self.y, y_pred, output_dict=True, zero_division=0)
        if plot:
            print(classification_report(self.y, y_pred, zero_division=0))

        self.scores[runtype]["accuracy"] = cr["accuracy"]
        self.scores[runtype]["precision"] = cr["weighted avg"]["precision"]
        self.scores[runtype]["recall"] = cr["weighted avg"]["recall"]
        self.scores[runtype]["f1-score"] = cr["weighted avg"]["f1-score"]
        # The wasserstein distance is not a metric
        self.scores[runtype]["wasserstein-distance"] = wasserstein_distance(
            self.y, y_pred
        )  # note that this does not depend on the assignment made above so we get the same
        # value with self.clusters as well
        self.scores[runtype]["runtime"] = self.runtime[runtype]
        self.scores[runtype]["adjusted-rand-index"] = adjusted_rand_score(
            self.y, y_pred
        )

        return disp, cr

    def plot_assignment_matrix(
        self, pred_labels=None, target_decoder=None, runtype=None, ax=None, show=True
    ):
        """
        Plot the assignment matrix
        :param pred_labels: predicted labels
        :param target_decoder: target decoder
        :return: None
        """
        if runtype is not None:
            assert runtype in self.all_clusters.keys()
            runtype = runtype
            clusters = self.all_clusters[runtype]
            pred_labels_default = self.all_pred_labels[runtype]
        elif self.clusters is None:
            raise Exception("Clusters not computed yet")
        else:
            runtype = self.last_runtype
            clusters = self.clusters
            pred_labels_default = self.pred_labels

        n_clusters = max(len(np.unique(clusters)), len(np.unique(self.y)))
        if pred_labels_default is None and pred_labels is None:
            pred_labels = {i: i for i in range(1, 1 + n_clusters)}
        else:
            pred_labels = pred_labels_default if pred_labels is None else pred_labels
        if self.target_decoder is None and target_decoder is None:
            true_labels = {i: i for i in range(1, 1 + n_clusters)}
        else:
            true_labels = (
                self.target_decoder if target_decoder is None else target_decoder
            )

        clusters_histograms = np.zeros((n_clusters, n_clusters))
        for pred, true in zip(clusters, self.y):
            clusters_histograms[int(pred) - 1, int(true) - 1] += 1

        if ax is None:
            plt.imshow(clusters_histograms)
            plt.yticks(
                np.arange(n_clusters),
                [pred_labels[i] for i in range(1, 1 + n_clusters)],
            )
            plt.xticks(
                np.arange(n_clusters),
                [true_labels[i] for i in range(1, 1 + n_clusters)],
            )
            plt.xlabel("True class")
            plt.ylabel("Predicted class")
            plt.title(f"Assignment matrix ({runtype})")
            plt.colorbar()
            for i in range(n_clusters):
                for j in range(n_clusters):
                    plt.text(
                        j,
                        i,
                        int(clusters_histograms[i, j]),
                        ha="center",
                        va="center",
                        color=(
                            "w"
                            if clusters_histograms[i, j]
                            < clusters_histograms.max() / 2.0
                            else "black"
                        ),
                    )
            if show:
                plt.show()
        else:
            ax.imshow(clusters_histograms)
            ax.set_yticks(
                np.arange(n_clusters),
                [pred_labels[i] for i in range(1, 1 + n_clusters)],
            )
            ax.set_xticks(
                np.arange(n_clusters),
                [true_labels[i] for i in range(1, 1 + n_clusters)],
            )
            ax.set_xlabel("True class")
            ax.set_ylabel("Predicted class")
            ax.set_title(f"Assignment matrix ({runtype})")
            for i in range(n_clusters):
                for j in range(n_clusters):
                    ax.text(
                        j,
                        i,
                        int(clusters_histograms[i, j]),
                        ha="center",
                        va="center",
                        color=(
                            "w"
                            if clusters_histograms[i, j]
                            < clusters_histograms.max() / 2.0
                            else "black"
                        ),
                    )
            return ax

    def plot_histograms(self, runtype=None, show=False):
        """
        Plot the histograms of the predicted and true labels
        after mathching the predicted labels with the true ones
        either using optimal transport if the matching is already computed
        or using sorting otherwise
        :return: None
        """
        if runtype is not None:
            assert runtype in self.all_clusters.keys()
            clusters = self.all_clusters[runtype]
            pred_labels_default = self.all_pred_labels[runtype]
            runtype = runtype
        elif self.clusters is None:
            raise Exception("Clusters not computed yet")
        else:
            runtype = self.last_runtype
            clusters = self.clusters
            pred_labels_default = self.pred_labels

        n_clusters = max(len(np.unique(clusters)), self.n_clusters)
        if pred_labels_default is None:
            pred_labels = {i: i for i in range(1, 1 + n_clusters)}
        else:
            pred_labels = pred_labels_default
        if self.target_decoder is None:
            true_labels = {i: i for i in range(1, 1 + n_clusters)}
        else:
            true_labels = self.target_decoder

        hist_pred_ordered = np.zeros(n_clusters)
        hist_pred = np.sum(
            clusters == np.arange(1, 1 + n_clusters).reshape(-1, 1), axis=1
        )
        hist_true = np.sum(
            self.y == np.arange(1, 1 + n_clusters).reshape(-1, 1), axis=1
        )
        pred_labels_inv = {v: k for k, v in pred_labels.items()}

        for k, v in self.target_decoder.items():
            hist_pred_ordered[k - 1] = hist_pred[pred_labels_inv[v] - 1]

        plt.bar(np.arange(n_clusters), hist_pred_ordered)
        plt.bar(np.arange(n_clusters), hist_true, alpha=0.5)
        plt.xticks(
            np.arange(n_clusters),
            [true_labels[i + 1] for i in range(n_clusters)],
        )
        plt.legend(["Predicted", "True"])
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        plt.title(f"Histograms ({runtype})")
        if show:
            plt.show()

    def plot_tsne(self, runtype=None, fig=None, ax=None, show=True):
        """
        Plot the t-SNE of the data in 2D
        If the clusters are already computed, plot the t-SNE of the data
        with the true labels and the predicted labels
        :return: None
        """
        if runtype is not None:
            assert runtype in self.all_clusters.keys()
            clusters = self.all_clusters[runtype]
        elif self.clusters is None:
            raise Exception("Clusters not computed yet")
        else:
            runtype = self.last_runtype
            clusters = self.clusters

        if "tsne" not in dir(self):
            self.tsne = TSNE(
                n_components=2, perplexity=10, n_jobs=-1, random_state=self.seed
            )
            self.X_embedded = self.tsne.fit_transform(self.X)
        X_embedded = self.X_embedded

        only_pred = False
        if ax is not None and type(ax) is not np.ndarray:
            only_pred = True
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        if not (only_pred):
            ax[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.y)
            ax[0].set_title("True labels")
            ax[0].set_xlabel("t-SNE 1")
            ax[0].set_ylabel("t-SNE 2")
        if clusters is not (None):
            if only_pred:
                ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters)
                ax.set_title(f"Predicted labels {runtype}")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
            else:
                ax[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters)
                ax[1].set_title(f"Predicted labels {runtype}")
                ax[1].set_xlabel("t-SNE 1")
                ax[1].set_ylabel("t-SNE 2")
        if fig is not None:
            fig.suptitle(f"TSNE ({runtype})")
        if show:
            plt.show()

    def plot_mds(self, runtype=None):
        """
        Plot the MDS of the data in 2D
        If the clusters are already computed, plot the MDS of the data
        with the true labels and the predicted labels
        :return: None
        """
        if runtype is not None:
            assert runtype in self.all_clusters.keys()
            clusters = self.all_clusters[runtype]
        elif self.clusters is None:
            raise Exception("Clusters not computed yet")
        else:
            runtype = self.last_runtype
            clusters = self.clusters

        figmds, axmds = plt.subplots(1, 2, figsize=(10, 5))
        mds = MDS(n_components=2, n_jobs=-1, normalized_stress="auto")
        X_embedded = mds.fit_transform(self.X)
        if clusters is not (None):
            axmds[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters)
            axmds[1].set_title("Predicted labels")
        axmds[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.y)
        axmds[0].set_title("True labels")
        figmds.suptitle(f"MDS ({runtype})")
        plt.show()

    def save_dataset(self, path):
        self.X = pd.DataFrame(self.X, columns=self.columns)
        self.X.to_csv(path, index=False)


class Animals(BaseDataset):
    def __init__(self, path, target_path, n_iter=100, eps=1e-1, silent=False, seed=0):
        super().__init__(path, n_iter=n_iter, eps=eps, silent=silent, seed=seed)
        self.path = path
        self.data = pd.read_csv(path)
        self.data_target = pd.read_csv(target_path)
        self.compute_Xy()
        self.compute_n_cat()

    def compute_Xy(self):
        ordinal_encoders = {}

        animal_names = self.data["animal_name"].values
        y = self.data["class_type"].values

        self.target_decoder = {
            i: target
            for i, target in enumerate(self.data_target["Class_Type"].values, 1)
        }
        self.target_decoder_inv = {
            target: i
            for i, target in enumerate(self.data_target["Class_Type"].values, 1)
        }

        self.data_processed = self.data.copy()
        for col in self.data.drop(["animal_name"], axis=1).columns:
            ordinal_encoders[col] = OrdinalEncoder()
            self.data_processed[col] = (
                ordinal_encoders[col]
                .fit_transform(self.data[col].values.reshape(-1, 1))
                .squeeze()
                + 1
            )

        self.y = self.data_processed["class_type"].values
        self.X = self.data_processed.drop(["animal_name", "class_type"], axis=1)
        self.columns = self.X.columns
        self.X = self.X.values.astype(int)
        self.n_clusters = len(np.unique(y))

    def compute_n_cat(self):
        X = self.X
        self.m = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])])


class CarEvaluation(BaseDataset):
    def __init__(self, path, n_iter=100, eps=1e-1, silent=False, seed=0):
        super().__init__(path, n_iter=n_iter, eps=eps, silent=silent, seed=seed)
        columns = [
            "buying_price",
            "maintenance_price",
            "number_of_doors",
            "number_of_seats",
            "luggage_boot_size",
            "safety_rating",
            "car_acceptability",
        ]
        self.data = pd.read_csv(path, names=columns)
        self.path = path
        self.compute_Xy()

    def compute_Xy(self, n=None):
        categories = {
            "buying_price": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
            "maintenance_price": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
            "number_of_doors": {"2": 1, "3": 2, "4": 3, "5more": 4},
            "number_of_seats": {"2": 1, "4": 2, "more": 3},
            "luggage_boot_size": {"small": 1, "med": 2, "big": 3},
            "safety_rating": {"low": 1, "med": 2, "high": 3},
            "car_acceptability": {"unacc": 1, "acc": 2, "good": 3, "vgood": 4},
        }

        self.target_decoder_inv = categories["car_acceptability"]
        self.target_decoder = {v: k for k, v in self.target_decoder_inv.items()}

        self.data_processed = self.data.copy()
        for col, mapping in categories.items():
            self.data_processed[col] = self.data_processed[col].map(mapping)
        self.data_processed.head()

        X = self.data_processed.drop(columns=["car_acceptability"])
        y = self.data_processed["car_acceptability"]

        self.columns = X.columns

        if n is None:
            n = len(X)

        self.m = np.array([len(np.unique(X[col])) for col in X.columns])
        self.n_clusters = y.unique().shape[0]

        self.X, self.y = X.to_numpy()[n:], y.to_numpy()[n:]

        self.m = np.array([len(X[col].unique()) for col in X.columns])
        self.n_clusters = y.nunique()

        self.X, self.y = X.to_numpy(), y.to_numpy()
        permutation = np.random.permutation(len(self.X))
        self.X, self.y = self.X[permutation][:n].astype(int), self.y[permutation][:n]


class HayesRoth(BaseDataset):
    def __init__(self, path, n_iter=100, eps=1e-1, silent=False, seed=0):
        super().__init__(path, n_iter=n_iter, eps=eps, silent=silent, seed=seed)
        self.path = path
        self.data = pd.read_csv(path)
        self.data = self.data.drop(columns=["name"])
        self.compute_n_cat()
        self.compute_Xy()

    def compute_Xy(self):
        X = self.data.drop(columns=["class"]).astype(int)
        y = self.data["class"]
        self.target = "class"

        self.columns = X.columns

        self.target_decoder = {v: k for k, v in enumerate(y.unique(), 1)}
        self.target_decoder_inv = {k: v for k, v in enumerate(y.unique(), 1)}

        self.m = np.array([len(X[col].unique()) for col in X.columns])
        self.n_clusters = y.nunique()

        self.X, self.y = X.to_numpy().astype(int), y.to_numpy()


class Caesarian(BaseDataset):
    def __init__(self, path, n_iter=100, eps=1e-1, silent=False, seed=0):
        super().__init__(path, n_iter=n_iter, eps=eps, silent=silent, seed=seed)
        self.path = path
        self.data = pd.read_csv(path)
        self.compute_Xy()

    def compute_Xy(self):
        # Quantize age into 4 bins
        self.data_processed = self.data.copy()

        self.data_processed["Age"] = pd.qcut(
            self.data_processed["Age"], 4, labels=False
        )
        self.data_processed[
            [
                "Age",
                "Delivery time",
                "Blood of Pressure",
                "Heart Problem",
                "Caesarian",
            ]
        ] += 1
        X = self.data_processed.drop(columns=["Caesarian"]).astype(int)
        y = self.data_processed["Caesarian"]
        self.target = "Caesarian"

        self.columns = X.columns

        self.target_decoder = {v: k for k, v in enumerate(y.unique(), 1)}
        self.target_decoder_inv = {k: v for k, v in enumerate(y.unique(), 1)}

        self.m = np.array([len(X[col].unique()) for col in X.columns])
        self.n_clusters = y.nunique()

        self.X, self.y = X.to_numpy().astype(int), y.to_numpy()


class NurserySchool(BaseDataset):
    def __init__(self, path, n_iter=100, eps=1e-1, silent=False, seed=0):
        super().__init__(path, n_iter=n_iter, eps=eps, silent=silent, seed=seed)
        self.path = path
        self.data = pd.read_csv(path)
        self.categories = {
            "parents": ["usual", "pretentious", "great_pret"],
            "has_nurs": ["proper", "less_proper", "improper", "critical", "very_crit"],
            "form": ["complete", "completed", "incomplete", "foster"],
            "children": ["1", "2", "3", "more"],
            "housing": ["convenient", "less_conv", "critical"],
            "finance": ["convenient", "inconv"],
            "social": ["nonprob", "slightly_prob", "problematic"],
            "health": ["recommended", "priority", "not_recom"],
            "target": [
                "not_recom",
                "recommend",
                "very_recom",
                "priority",
                "spec_prior",
            ],
        }

        self.compute_Xy()

    def compute_Xy(self):
        df_nursery = self.data

        y = df_nursery["target"]
        df_X = df_nursery.drop("target", axis=1)

        categories_dict = {
            k: {v: i + 1 for i, v in enumerate(self.categories[k])}
            for k in self.categories
            if k != "target"
        }

        X = df_X.replace(categories_dict)
        self.target_decoder = {
            i + 1: val for i, val in enumerate(self.categories["target"])
        }
        self.target_decoder_inv = {
            val: i + 1 for i, val in enumerate(self.categories["target"])
        }
        y = y.replace(self.target_decoder_inv)

        self.columns = df_X.columns

        n = len(df_X)

        self.m = np.array([len(np.unique(X[col])) for col in X.columns])
        self.n_clusters = y.unique().shape[0]

        self.n_clusters = y.nunique()
        self.X, self.y = X.to_numpy()[n:], y.to_numpy()[n:]

        self.m = np.array([len(X[col].unique()) for col in X.columns])

        self.X, self.y = X.to_numpy(), y.to_numpy()
        permutation = np.random.permutation(len(self.X))
        self.X, self.y = self.X[permutation], self.y[permutation]
