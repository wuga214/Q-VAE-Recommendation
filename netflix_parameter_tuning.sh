#!/usr/bin/env bash
python tune_parameters.py -d data/netflix/ -n netflix/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/ifvae.csv -y config/ifvae.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/wrmf.csv -y config/wrmf.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/puresvd.csv -y config/pursvd.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/nceplrec.csv -y config/nceplrec.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/plrec.csv -y config/plrec.yml -gpu