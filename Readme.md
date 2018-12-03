Inference Friendly Variational Auto-encoder for Recommendation
==============================================================
![](https://img.shields.io/badge/linux-ubuntu-red.svg)
![](https://img.shields.io/badge/Mac-OS-red.svg)

![](https://img.shields.io/badge/cuda-8.0-green.svg)
![](https://img.shields.io/badge/python-2.7-green.svg)

![](https://img.shields.io/badge/cython-0.28.5-blue.svg)
![](https://img.shields.io/badge/cupy-4.0.0-blue.svg)
![](https://img.shields.io/badge/scipy-1.0.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.14.1-blue.svg)
![](https://img.shields.io/badge/sklearn-0.19.0-blue.svg)
![](https://img.shields.io/badge/pandas-0.20.3-blue.svg)
![](https://img.shields.io/badge/tqdm-4.11.2-blue.svg)
![](https://img.shields.io/badge/argparse-1.1-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.4.0-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)
![](https://img.shields.io/badge/fbpca-1.0-blue.svg)
![](https://img.shields.io/badge/pyyaml-4.1-blue.svg)

Please Disable GPU usage in `main.py` if needed.


# Algorithm Implemented
1. Inference Friendly Variational Autoencoder(IFVAE)
2. Variational Autoencoder for Collaborative Filtering(VAE-CF)
3. Collaborative Metric Learning(CML)
4. Auto-encoder Recommender(AutoRec)
5. Collaborative Denoising Auto-Encoders(CDAE)
6. Weighted Regularized Matrix Factorization(WRMF)
7. Pure SVD Recommender(PureSVD)
8. Bayesian Personalized Ranking(BPR)
9. Popularity

# Data
1. Movielens 1M,
2. Movielens 20M,
3. Yahoo 1R,
4. Netflix,
5. Amazon Prize

Data is not suit to submit on github, so please prepare it yourself. It should be numpy npy file directly
dumped from csr sparse matrix. It should be easy..

# Measure
The above algorithm could be splitted into two major category based on the distance
measurement: Euclidean or Cosine. CML is a euclidean distance recommender. And, ALS
is a typical Cosine distance recommender. When doing evaluation, please select
similarity measurement before running with `--similarity Euclidean`

# Example Commands

### Single Run
```
python main.py -d datax/ -m VAE-CF -i 200 -l 0.0000001 -r 100
```

### Hyper-parameter Tuning
Please checkout the `tune_parameters.py` file to specific which algorithm and which hyper-parameter to tune.
```
python tune_parameters.py
```

### Reproduce Paper Results
```
python reproduce_paper_results.py
```


