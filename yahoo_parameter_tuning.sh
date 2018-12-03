#!/usr/bin/env bash
python tune_parameters.py -d data/yahoo/ -n yahoo/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/ifvae.csv -y config/ifvae.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/wrmf.csv -y config/wrmf.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/puresvd.csv -y config/puresvd.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/nceplrec.csv -y config/nceplrec.yml -gpu
python tune_parameters.py -d data/yahoo/ -n yahoo/plrec.csv -y config/plrec.yml -gpu