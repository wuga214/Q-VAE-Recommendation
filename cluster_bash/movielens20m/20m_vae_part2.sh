#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/movielens20m/ -n movielens20m/vae-part2.csv -y config/vae-part2.yml
