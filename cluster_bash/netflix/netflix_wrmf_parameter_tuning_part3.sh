#!/usr/bin/env bash
source ~/ENV/bin/activate
cd IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/wrmf-part3.csv -y config/wrmf-part3.yml