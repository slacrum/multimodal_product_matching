# Similarity Learning of Product Descriptions and Images using Multimodal Neural Networks

This is the code implementation of "Similarity Learning of Product Descriptions and Images using Multimodal Neural Networks", submitted at the [Natural Language Processing Journal by Elsevier](https://www.sciencedirect.com/journal/natural-language-processing-journal).

# Project structure

    .
    ├──assets                           # images, docs and other supplementary resources
    |
    ├──datasets                         # default location for datasets
    │  └──abo                           # Amazon Berkeley Objects (ABO) dataset
    │
    ├──data_loader                      # data loading and preprocessing
    │  ├──dataset.py                    # abstract Dataset class (handles download and overall preprocessing)
    │  └──abo.py                        # ABO class (inherits Dataset), also serves as template for custom data
    │
    ├──models                           # Model implementations in Tensorflow/Keras
    │  ├──char_cnn_zhang.py             # Character-level CNN by Zhang et al. (2015)
    │  └──mnn_em.py                     # MNN-EM (base + extended model)
    |
    ├──notebooks                        # Jupyter notebooks
    │  ├──experiments                   # Experiments (from downloading, preprocessing up to training and evaluation)
    |  |  ├──configs                    # JSON configs for data (path, alphabet size), models (layer shapes) and training (hyperparams, callbacks, metrics)
    |  |  ├──mnn_em.ipynb               # MNN-EM experiments
    |  |  └──extended_mnn_em.ipynb      # Extended MNN-EM experiments
    |  ├──data_preparation.ipynb        # "Hands-on" ABO Data preparation, covering every processing step handled by the Data Loader
    |  └──visualize_results.ipynb       # Visualize "advanced" metrics, such as ROC and Precision-Recall curve and threshold optimization
    |
    ├──runs                             # default location for experiment outputs
    |
    ├──utils                            # Utilities and convenience functions
    │  ├──eval.py                       # Evaluation-related functions (generate callbacks, plot and save metrics,...)
    │  ├──img_processing.py             # Generate image embeddings with CNNs
    │  └──text_processing.py            # Tokenize texts and create embedding weights
    │
    ├──.gitignore
    ├──README.md                        # this very file
    └──requirements.txt                 # Pip requirements

# Installation
```
pip install -r requirements.txt
```

# Get started
In general, all the notebooks provided are standalone and may be run separately. However, we still recommend the following order:
1. [Data Preparation](./notebooks/data_preparation.ipynb), to gain understanding of the ABO data and necessary processing
2. Experiments for running our ML pipeline:
    1. [Base MNN-EM](./notebooks/experiments/mnn_em.ipynb)
    2. [Extended MNN-EM](./notebooks/experiments/extended_mnn_em.ipynb)
3. [Visualizing results](./notebooks/visualize_results.ipynb), such as ROC curve, Precision Recall curve and perform threshold optimization