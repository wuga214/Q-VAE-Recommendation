#!/usr/bin/env bash
python tune_parameters.py -d data/movielens20m/ -n movielens20m/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/ifvae.csv -y config/ifvae.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/wrmf.csv -y config/wrmf.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/puresvd.csv -y config/pursvd.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/nceplrec.csv -y config/nceplrec.yml -gpu
python tune_parameters.py -d data/movielens20m/ -n movielens20m/plrec.csv -y config/plrec.yml -gpu