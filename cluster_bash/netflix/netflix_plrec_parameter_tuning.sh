#!/usr/bin/env bash
source ~/ENV/bin/activate
cd IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/plrec.csv -y config/plrec.yml
