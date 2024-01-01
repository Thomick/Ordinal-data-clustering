import numpy as np
from sklearn.cluster import KMeans
try:
    from src.compute_u import compute_u
    from src.god_model_estimator import estimate_mu_pi as estimate_mu_pi_god
    from src.bos_model_estimator import univariate_em as estimate_mu_pi_bos
except ImportError:
    from compute_u import compute_u
    from god_model_estimator import estimate_mu_pi as estimate_mu_pi_god
    from bos_model_estimator import univariate_em as estimate_mu_pi_bos


class AECM:
    """
    AECM algorithm for clustering

    Attributes:
        nb_clusters, int: 
            number of clusters
        nb_features, int: 
            number of features
        ms, np.ndarray (nb_features,) of int:
            number of categories for each feature
        alphas, np.ndarray (nb_clusters,) of float:
            mixing coefficients of the clusters
        mus, np.ndarray (nb_clusters, nb_features) of int:
            positions parameters of each cluster and each feature
        pis, np.ndarray (nb_clusters, nb_features) of float:
            precision parameters of each cluster and each feature
        log_likelihood, float:
            log likelihood of the model
        bic, float:
            BIC of the model
        data, np.ndarray (nb_samples, nb_features) of int:
            data to cluster (data[., j] in [[1, ms[j] ]])
        univariate_likelihoods, np.ndarray (max ms, nb_clusters, nb_features) of float:
            likelihoods of each value for each feature and each cluster
            univariate_likelihoods[x, k, j] = P(x[j] | mu[k, j], pi[k, j])
            and univariate_likelihoods[x, k, j] = 1 if x[j] > ms[j]
        px_knowing_w, np.ndarray (nb_samples, nb_clusters) of float:
            probability of each sample knowing the cluster
            px_knowing_w[i, k] = P(x^i | wik = 1, mu, pi)
        pwik, np.ndarray (nb_samples, nb_clusters) of float:
            probability of each sample to belong to each cluster
            P(wik = 1 | x^i, alpha, mu, pi)
        pxi, np.ndarray (nb_samples) of float:
            probability of each sample
            P(x^i | alpha, mu, pi)
        eps, float:
            threshold to stop the algorithm (difference between two log likelihoods)
        seed, int:
            seed for the initialization of the KMeans algorithm
        verbose, bool:
            whether to print intermediate results
    """
    def __init__(self,
                 nb_clusters: int,
                 nb_features: int,
                 ms: np.ndarray,
                 data: np.ndarray,
                 eps: float = 1e-5,
                 seed: int = 0,
                 verbose: bool = False):
        """
        Args:
            nb_clusters, int: 
                number of clusters
            nb_features, int: 
                number of features
            ms, np.ndarray (nb_features,) of int:
                number of categories for each feature
            data, np.ndarray (nb_samples, nb_features) of int:
                data to cluster (data[., j] in [[1, ms[j] ]])
            verbose, bool:
                whether to print intermediate results
        """
        self.nb_clusters = nb_clusters
        self.nb_features = nb_features
        self.ms = ms
        self.alphas = np.ones(nb_clusters) / nb_clusters
        self.mus = np.ones((nb_clusters, nb_features), dtype=int)
        self.pis = np.ones((nb_clusters, nb_features))
        self.log_likelihood = -np.inf
        self.data = data
        self.eps = eps
        self.seed = seed
        self.verbose = verbose
        assert self.data.shape[1] == self.nb_features, f"Data has {self.data.shape[1]} features but {self.nb_features} were expected"
        assert self.ms.shape[0] == self.nb_features, f"ms has {self.ms.shape[0]} features but {self.nb_features} were expected"
        assert self.data.min(axis=1) >= 1, f"Data has values less than 1 for each feature"
        assert data.max(axis=1) <= self.ms, f"Data has values greater than the number of categories for each feature"
        self.pwik = np.ones((self.data.shape[0], self.nb_clusters))
        self.pxi = np.ones(self.data.shape[0])
        self.univariate_likelihoods = np.ones((self.ms.max(), self.nb_clusters, self.nb_features))
        self.px_knowing_w = np.ones((self.data.shape[0], self.nb_clusters))

    @property
    def bic(self) -> float:
        """
        Compute the BIC of the model
        """
        return self.log_likelihood - 0.5 * (self.nb_clusters - 1 + self.nb_features * self.nb_clusters) * np.log(self.data.shape[0])

    def _init_k_means(self):
        """
        Initialize the parameters of the model using the KMeans algorithm
        """
        kmeans = KMeans(n_clusters=self.nb_clusters, random_state=self.seed).fit(self.data)
        self.alphas = np.bincount(kmeans.labels_) / self.data.shape[0]
        for k in range(self.nb_clusters):
            for j in range(self.nb_features):
                self.mus[k, j] = np.floor(np.mean(self.data[kmeans.labels_ == k, j]))
                self.pis[k, j] = np.mean(self.data[kmeans.labels_ == k, j] == self.mus[k, j])
    
    def _init_random(self):
        """
        Initialize the parameters of the model randomly
        """
        self.alphas = np.random.dirichlet(np.ones(self.nb_clusters))
        self.pis = np.random.random((self.nb_clusters, self.nb_features))
        self.mus = np.random.randint(1, self.ms + 1, (self.nb_clusters, self.nb_features))

    def univariate_mu_pi_estimation(self, m, data, weiths, **kwargs) -> tuple[int, float, np.ndarray]:
        """
        Estimate mu, pi for 1D data

        Arguments:
        ----------
            m, int: number of categories
            data: data to cluster (data[., j] in [[1, m]])
            weights: weights of the data
            **kwargs: arguments for the estimation of mu, pi
        
        Return:
        -------
            mu, int: position parameter
            pi, float: precision parameter
            probs, np.ndarray (m,) of float: probabilities of each value
                [ P(x | mu, pi) for x in [[1, m]] ]
        """
        raise NotImplementedError

    def _internal_estimation(self, **kwargs):
        """
        Run the internal estimation of the parameters of the model
        for each feature and each cluster
        
        Args:
            weigths:
                
            **kwargs: arguments for the internal estimation
        """
        for k in range(self.nb_clusters):
            weights = self.pwik[:, k]
            for j in range(self.nb_features):
                self.mus[k, j], self.pis[k, j], probs = self.univariate_mu_pi_estimation(
                    m=self.ms[j], data=self.data[:, j], weights=weights, **kwargs)
                self.univariate_likelihoods[:, k, j] = probs

    def _maximization_step(self):
        """
        Run the maximization step of the AECM algorithm:
        alpha_k = 1 / n * sum_i^n P(wik = 1 | x^i, alpha, mu, pi)
        """
        self.alphas = np.mean(self.pwik, axis=0)
    
    # WARNING: TO CHECK
    def _expectation_step(self):
        """
        Run the expectation step of the AECM algorithm:
        compute P(wik = 1 | x^i, alpha, mu, pi) stored in self.pwik
        compute P(x^i | alpha, mu, pi) stored in self.pxi
        use the univariate likelihoods stored in self.univariate_likelihoods
        """
        # px_knowing_w[i, k] = P(x^i | wik = 1, mu, pi)
        for i in range(self.data.shape[0]):
            self.px_knowing_w[i] = np.prod(self.univariate_likelihoods[self.data[i] - 1, :, np.arange(self.nb_features)], axis=0)
        self.pxi = np.sum(self.px_knowing_w * self.alphas, axis=1)
        self.pwik = self.px_knowing_w * self.alphas / self.pxi[:, None]
    
    def _compute_loglikelihood(self):
        """
        Compute the current log likelihood of the model
        """
        self.log_likelihood = np.sum(np.log(self.pxi))
    
    def fit(self, initialization: str = "kmeans", **kwargs) -> list[float]:
        """
        Run the AECM algorithm

        Arguments:
        ----------
            initialization, str in {'kmeans', 'random'}:
                initialization method for the parameters of the model
        """
        if initialization == "kmeans":
            self._init_k_means()
        elif initialization == "random":
            self._init_random()
        else:
            raise AttributeError(f"{initialization=} should be either kmeans or random")
    
        self._internal_estimation(**kwargs)
        old_log_likelihood = self.log_likelihood + 2 * self.eps
        log_likelihoods = [old_log_likelihood]

        while abs(old_log_likelihood - self.log_likelihood) > self.eps:
            old_log_likelihood = self.log_likelihood
            self._expectation_step()
            self._maximization_step()
            self._internal_estimation(**kwargs)
            self._compute_loglikelihood()
            if self.verbose:
                print(f"log likelihood: {self.log_likelihood}")
            log_likelihoods.append(self.log_likelihood)
        return log_likelihoods


class AECM_GOD(AECM):
    """
    AECM algorithm for clustering with the God model

    Attributes:
        All attributes of AECM

        u, dict[int, np.ndarray (m, m, m)]:
            u[m] : u(., ., .) coefficients of the polynomials (m, m, m)
    """
    def __init__(self,
                 nb_clusters: int,
                 nb_features: int,
                 ms: np.ndarray,
                 data: np.ndarray,
                 eps: float = 1e-5,
                 seed: int = 0,
                 verbose: bool = False):
        """
        Args:
            nb_clusters, int: 
                number of clusters
            nb_features, int: 
                number of features
            ms, np.ndarray (nb_features,) of int:
                number of categories for each feature
            data, np.ndarray (nb_samples, nb_features) of int:
                data to cluster (data[., j] in [[1, ms[j] ]])
            verbose, bool:
                whether to print intermediate results
        """
        super().__init__(nb_clusters, nb_features, ms, data, eps, seed, verbose)
        self.u = dict()
        for m in ms:
            if m not in self.u:
                self.u[m] = compute_u(m)

    def univariate_mu_pi_estimation(self, 
                                    m: int, 
                                    data: np.ndarray, 
                                    weiths: np.ndarray,
                                    epsilon: float) -> tuple[int, float, np.ndarray]:
        """
        Estimate mu, pi for 1D data

        Arguments:
        ----------
            m, int: number of categories
            data: data to cluster (data[., j] in [[1, m]])
            weights: weights of the data
            epsilon, float: precision threshold on the value of pi
        
        Return:
        -------
            mu, int: position parameter
            pi, float: precision parameter
            probs, np.ndarray (m,) of float: probabilities of each value
                [ P(x | mu, pi) for x in [[1, m]] ]
        """
        mu, pi, _, probs = estimate_mu_pi_god(m=m, data=data, weights=weiths, epsilon=epsilon, u=self.u[m])
        return mu, pi, probs


class AECM_BOS(AECM):
    def univariate_mu_pi_estimation(self, 
                                    m: int, 
                                    data: np.ndarray, 
                                    weiths: np.ndarray,
                                    n_iter: int = 100,
                                    epsilon: float = 1e-3,
                                    pi_start: float = 0.5,
                                    ) -> tuple[int, float, np.ndarray]:
        """
        Estimate mu, pi for 1D data

        Arguments:
        ----------
            m, int: number of categories
            data: data to cluster (data[., j] in [[1, m]])
            weights: weights of the data
            n_iter: int
                number of iterations of the EM algorithm
            eps: float
                precision threshold on the value of pi
            pi_start, float:
                pi to start
        Return:
        -------
            mu, int: position parameter
            pi, float: precision parameter
            probs, np.ndarray (m,) of float: probabilities of each value
                [ P(x | mu, pi) for x in [[1, m]] ]
        """
        mu, pi, _, probs = estimate_mu_pi_bos(m=m, 
                                              data=data, 
                                              weights=weiths,
                                              n_iter=n_iter,
                                              eps=epsilon,
                                              pi=pi_start)
        return mu, pi, probs
