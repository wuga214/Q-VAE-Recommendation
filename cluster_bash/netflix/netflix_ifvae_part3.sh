#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/ifvae-part3.csv -y config/ifvae-part3.yml

