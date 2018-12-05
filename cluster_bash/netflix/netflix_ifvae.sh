#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/ifvae.csv -y config/ifvae.yml
