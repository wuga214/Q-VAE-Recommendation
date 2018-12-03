#!/usr/bin/env bash
python getnetflix.py --implicit -r 0.5,0.2,0.3 -d data/netflix/ -f data/netflix/raw/training_set
python tune_parameters.py -d data/netflix/ -n netflix/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/ifvae.csv -y config/ifvae.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/wrmf.csv -y config/wrmf.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/puresvd.csv -y config/puresvd.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/nceplrec.csv -y config/nceplrec.yml -gpu
python tune_parameters.py -d data/netflix/ -n netflix/plrec.csv -y config/plrec.yml -gpu

python getnetflix.py --implicit -r 0.7,0.3,0.0 -d data/netflix/ -f data/netflix/raw/training_set
python reproduce_paper_results.py -d data/netflix/ -v Rvalid.npz -n netflix_test_result.csv -gpu