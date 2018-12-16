#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/yahoo/ -n yahoo/wrmf-part14.csv -y config/wrmf-part14.yml
