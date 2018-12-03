#!/usr/bin/env bash
python getmovielens.py --implicit -r 0.5,0.2,0.3 -d datax/ml-1m/ -n ratings.csv
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/autorec.csv -y config/autorec.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/bpr.csv -y config/bpr.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/cdae.csv -y config/cdae.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/cml.csv -y config/cml.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/ifvae.csv -y config/ifvae.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/vae.csv -y config/vae.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/wrmf.csv -y config/wrmf.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/puresvd.csv -y config/puresvd.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/nceplrec.csv -y config/nceplrec.yml -gpu
python tune_parameters.py -d datax/ml-1m/ -n movielens1m/plrec.csv -y config/plrec.yml -gpu

python getmovielens.py --implicit -r 0.7,0.3,0.0 -d datax/ml-1m/ -n ratings.csv
python reproduce_paper_results.py -d datax/ml-1m/ -v Rvalid.npz -n movielens1m_test_result.csv -gpu