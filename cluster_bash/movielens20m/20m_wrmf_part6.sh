#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/movielens20m/ -n movielens20m/wrmf-part6.csv -y config/wrmf-part6.yml