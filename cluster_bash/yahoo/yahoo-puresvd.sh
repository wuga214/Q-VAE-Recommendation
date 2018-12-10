#!/usr/bin/env bash
source ~/ENV/bin/activate
cd /media/wuga/Experiments/SIGIR-19/IF-VAE-Recommendation
python tune_parameters.py -d data/yahoo/ -n yahoo/puresvd.csv -y config/puresvd.yml
