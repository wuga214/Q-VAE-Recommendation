#!/usr/bin/env bash
source ~/ENV/bin/activate
cd IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/cdae.csv -y config/cdae.yml
