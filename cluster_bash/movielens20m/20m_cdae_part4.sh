#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/movielens20m/ -n movielens20m/cdae-part4.csv -y config/cdae-part4.yml
