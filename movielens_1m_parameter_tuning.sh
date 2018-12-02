#!/usr/bin/env bash
python tune_parameters.py -n movielens1m/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -n movielens1m/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -n movielens1m/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -n movielens1m/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -n movielens1m/ifvae.csv -y config/ifvae.yml -gpu
python tune_parameters.py -n movielens1m/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -n movielens1m/wrmf.csv -y config/wrmf.yml -gpu