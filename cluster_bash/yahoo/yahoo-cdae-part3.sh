#!/usr/bin/env bash
source ~/ENV/bin/activate
cd /media/wuga/Experiments/SIGIR-19/IF-VAE-Recommendation
python tune_parameters.py -d data/yahoo/ -n yahoo/cdae-part3.csv -y config/cdae-part3.yml
