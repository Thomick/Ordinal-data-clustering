# Ordinal-data-clustering

This repo contains experiments and analysis of ordinal data clustering methods based on the paper by Biernacki et al. (2015). The paper can be found [here](https://inria.hal.science/hal-01052447v2/document).

## Installation

In order to run the code and the notebooks, you need to run the following commands:

```bash
git clone https://github.com/Thomick/Ordinal-data-clustering
cd Ordinal-data-clustering
pip install -r requirements.txt
```

## Usage

The notebooks can be run using Jupyter Notebook or Jupyter Lab and are located in the `notebooks` folder. The source code is located in the `src` folder.

In order to use to cluster datasets using the clustering algorithms implemented in this repo (BOS distribution and GOD model), you can use the following code:

```python
from src.clustering import OrdinalClustering
import numpy as np

# Load the dataset
data = ...
m = [np.unique(data[:, i]).shape[0] for i in range(data.shape[1])]

# Create the clustering object
oc = OrdinalClustering(n_clusters=3, method='bos', init="random") 
# or method='god', init="random" or init="kmeans"

# Fit the clustering object to the data
clusters = oc.fit_transform(data)
```

It is also possible to extend the BaseDataset class in order to create a custom dataset and plot or analyze it using predefined methods. For example, for the Zoo dataset, we can do the following:

```python
from src.dataset import Animals

dataset = Animals(data_path="./data/zoo.csv")

dataset.cluster_bos(init="random") # dataset.cluster_god(), dataset.cluster_kmeans(), dataset.cluster_gaussian()
dataset.plot_tsne()
dataset.classification_results()
dataset.plot_assignment_matrix()
dataset.plot_histograms()
```

## Structure

The repo is structured as follows:
- `notebooks`: Contains the notebooks used for the experiments and analysis that are available in the report.
- `src`: Contains the source code for the estimation algorithms and the clustering algorithms.
    - `src/data_generator.py`: Contains the code for generating synthetic datasets.
    - `src/clustering.py`: Contains the code for the estimation EM and AECM algorithms (in the univariate and the multivariate case) for both the BOS and the GOD distribution models.
    - `src/dataset.py`: Contains the code for loading the datasets and analyzing the different methods on them for generating the results on the report.
- `data`: Contains the datasets used for the experiments.
- `report`: Contains the report for the project.

## Data
The datasets used in the notebooks are located in the `data` folder. Here the datasets used for the expeeriments (references for the datasets can be found in the report):
- `data/car_evaluation.csv`: Car Evaluation Database.
- `data/hayes-roth.csv`: Hayes-Roth Database.
- `data/zoo.csv`: Zoo Database.

Moreover, the estimation algorithms are tested on synthetic datasets generated specifically for the respective distributions (BOS distribution and GOD distribution).

