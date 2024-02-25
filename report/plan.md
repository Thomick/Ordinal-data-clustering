# Plan du rapport:
## Intro:
- motivation
- état de l'art
- contribution par rapport au papier original
- plan

## Méthodes:
- description AECM
- description BOS
    - propriétés de BOS
    - description EM pour BOS
    - description trichotomy (principe, preuve, complexité...)
    - decription trichotomy pour BOS
- description de GOD
    - trichotomy pour GOD

## Expériences:
- comparaison temps de calcul EM-BOS, trichotomy-BOS, GOD
- comparaison des méthodes sur des données synthétiques
- comparaison des méthodes sur des données réelles


## Conclusion:

## Bibliographie

## Annexes

- détails des algorithmes
- détails des expériences


# Trichotomy:

## Setting:

Let suppose that we have a set of $n$ independant observations $X = (x_i)_{i \in [n]}$, wehre $x_i \in [[1, m]]$ follow a distribution $P$ with parameters $\mu, \pi$ with $\mu \in [[1, m]]$ and $\pi \in [[a, b]]$. We want to estimate $\mu$ and $\pi$. We choose the estimate that maximize the likelihood of the data.
$$(\mu, \pi) = \argmax_{(\mu, \pi) \in [[1, m]] \times [[a, b]]} P(X | \mu, \pi)$$

As the data are independant, we have:
$$P(X | \mu, \pi) = \prod_{i=1}^n P(x_i | \mu, \pi)$$

As the number of possible values for $x$ is finite, we can group the data by values and count the number of occurences of each value. Let $n_i$ be the number of occurences of $i$ in the data. We have:
$$P(X | \mu, \pi) = \prod_{i=1}^m P(i | \mu, \pi)^{n_i}$$

In the AECM algorithm, each data point has a weight $w_i$. As previously, we can suppose without loss of generality that we have only one observation of each value with a specific weight (we can always group the data by values and sum the weights of the observations). With the weights $W \in \mathbb{R}_+^n$ where $w_i$ is the weight of the value $i$, we can write the weighted likelihood as:

$$P(W | \mu, \pi) = \prod_{i=1}^m P(i | \mu, \pi)^{w_i}$$

We can also write the weighted $\log$-likelihood as:

$$L_W(\mu, \pi) := \log P(W | \mu, \pi) = \sum_{i=1}^m w_i \log P(i | \mu, \pi)$$


## Optimization: 

To estimate $\mu$ and $\pi$, the idea proposed for the BOS-model in REF is to use the Expectaion-Maximization algorithm. However they note that it is easier to first estimate $\pi$ for every possible value of $\mu$ and then to estimate $\mu$ using the estimated $\pi$. In forulas, we have:


$$\hat{\pi_{\mu}} = \argmax_{\pi \in [[a, b]]} P(W | \mu, \pi)$$
$$\hat{\mu} = \argmax_{\mu \in [[1, m]]} \max_{\pi \in [[a, b]]} P(W | \mu, \pi) = \argmax_{\mu \in [[1, m]]} P(W | \mu, \hat{\pi_{\mu}})$$

Once we have the estimates $\hat{\pi_{\mu}}$ for every possible value of $\mu$, it is easy to estimate $\mu$ by choosing the value of $\mu$ that maximize the weighted likelihood as it requires only to compute the likelihood for every possible value of $\mu$ and to choose the maximum.

Estimating $\pi$ for a given $\mu$ is a one-dimensional optimization problem. We can use the EM algorithm to solve it but it may be possible to use a direct optimization algorithm. For example if the function $\pi \mapsto L_W(\mu, \pi)$ is stricly concave (or constant), we can the ternary search algorithm (or trisection algorithm) to find the maximum.

### Concavity of the weighted likelihood:

Suppose we have:

$$\forall x \in [[1, m]], \pi \mapsto P(x | \mu, \pi) \text{ is strcictly $\log$-concave}$$

Then we have, $\pi \mapsto L_W(\mu, \pi)$ is stricly concave as positive linear combination of concave functions are concave.

### Ternary search algorithm:

TODO (see https://en.wikipedia.org/wiki/Ternary_search)

### Complexity:

We note $n$ the numbre of observations, $m$ the number of possible values for $x$ and $\epsilon > 0$ the precision of the estimate of $\pi$.

We can first group the data by values and sum the weights of the observations. This can be done in $\theta(n)$ operations.

Then to estimate $\pi$ for a given $\mu$, we can use the ternary search algorithm. The number of iterations of the algorithm is $O(\log \frac{b - a}{\epsilon})$. For each iteration, we need to compute the likelihood for two values of $\pi$. If we note $C_E(m)$ the complexity of computing the likelihood for a given value of $\pi$, we have a complexity of $O(\log \frac{b - a}{\epsilon} C_E(m))$.

Finally, to estimate $\mu$, we need to compute the likelihood for every possible value of $\mu$. This gives a total complexity of $O(m C_E(m) \log \frac{b - a}{\epsilon} )$.

In our case we will have $C_E(m) = \theta(m)$ and $[a, b] \subset [0, 1]$ which gives a complexity of $O(m^2 \log \frac{1}{\epsilon})$.
