# Clustering Multivariate Ordinal Data

This repository contains the code associated to the report Clustering Multivariate Ordinal Data submited to [IPOL](https://www.ipol.im/). 

The authors of this project are Théo Rudkiewicz, Ali Ramlaoui and Thomas Michel.

The present work reviews and largely extends the approach proposed in the paper : Christophe Biernacki and Julien Jacques. Model-based clustering of multivariate ordinal data relying on a stochastic binary search algorithm. Statistics and Computing, 26:929–943, 2016. Link to the paper [here](https://inria.hal.science/hal-01052447v2/document).

## Installation

In order to run the code and the notebooks, you need to run the following commands:

```bash
git clone https://github.com/Thomick/Ordinal-data-clustering
cd Ordinal-data-clustering
pip install -r requirements.txt
```

You can also use it online, see : https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000487


## Structure

The repo is structured as follows:
- `notebooks`: Contains the notebooks used for the experiments and analysis that are available in the report.
- `src`: Contains the source code for the estimation algorithms and the clustering algorithms.
    - `src/data_generator.py`: Contains the code for generating synthetic datasets.
    - `src/clustering.py`: Contains the code for the estimation EM and AECM algorithms (in the univariate and the multivariate case) for both the BOS and the GOD distribution models.
    - `src/dataset.py`: Contains the code for loading the datasets and analyzing the different methods on them for generating the results on the report.
- `data`: Contains the datasets used for the experiments.
- `report`: Contains the report for the project.

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

## Data and experiments
The datasets used in the notebooks are located in the `data` folder. Here the datasets used for the experiments (licenses for the real-life datasets can be found below in the corresponding sections) can be separated in different categories. We detail the generation process and the way they were used for the experiments for Reproducibility purposes.

### Synthetic datasets
For the runtime comparisons of our implementation of the AECM algorithm and the original implementation (ordinalClust), the datasets in `data/comparison_curves`. These datasets were generated using the notebook `notebooks/experiments_synthetic.ipynb` and the plots for the report are made on the same notebook. The seeds are specified in the dataset file (and are exactly the same that are in the notebook currently for both the algorithms estimation and the datasets). The R package ordinalClust is used in the scripts on the folder `R/` and uses the exact same datasets for the comparison. Moreover, we save the results of the runtimes on the datasets for every value of the number of categories ($m$) and every different curve

The other experiments on synthetic datasets are also available on the notebook `notebooks/experiments_synthetic.ipynb` and `notebooks/estimation.ipynb`. They use data that is generated on the fly for the experiments but since the seeds are fixed the experimences are also reproducible by just running the notebooks. We also save the table of the results of the experiments in the `data/synthetic/` folder.

### Real-life datasets
The real-life datasets are detailed in the report but can be found in the locations specified in the license. We also put the processed datasets in the `data/processed/` folder that can directly be used for clustering. The experiments on these datasets can be found in the notebook `notebooks/clustering_datasets.ipynb` and they are all reproducible thanks to the fixed seed. The plots and the tables found in the report are all available on the same notebook.

## License

The authors of this project are Théo Rudkiewicz, Ali Ramlaoui and Thomas Michel.

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.
This includes all the code and the report but not the datasets used for the experiments.

- `data/car_evaluation.csv`: Car Evaluation Database created by Marko Bohanec (https://archive.ics.uci.edu/dataset/19/car+evaluation)
- `data/hayes-roth.csv`: Hayes-Roth Database created by Barbara Hayes-Roth and Frederick Hayes-Roth (https://archive.ics.uci.edu/dataset/44/hayes+roth)
- `data/zoo.csv`: Zoo Database created by Richard Forsyth (https://archive.ics.uci.edu/dataset/111/zoo)
- `data/nursery.csv`: Nursery Database created by Vladislav Rajkovic (https://archive.ics.uci.edu/dataset/76/nursery)
- `data/caesarian.csv`: Caesarian Database created by Muhammad Amin and Amir Ali (https://archive.ics.uci.edu/dataset/472/caesarian+section+classification+dataset)
- `data/aeres.csv`: Aeres Database created by Biernacki et al. (2015)
