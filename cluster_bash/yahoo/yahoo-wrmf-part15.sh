#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/yahoo/ -n yahoo/wrmf-part15.csv -y config/wrmf-part15.yml
