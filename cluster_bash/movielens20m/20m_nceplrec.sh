#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/movielens20m/ -n movielens20m/nceplrec.csv -y config/nceplrec.yml
