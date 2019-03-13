#!/bin/sh

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-sample-all -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-latent -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-iterative -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-sample-all --disable-latent -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-sample-all --disable-iterative -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-latent --disable-iterative -alm ThompsonSampling

python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 50 -k 1 --disable-gpu --disable-sample-all --disable-latent --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-sample-all -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-latent -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-sample-all --disable-latent -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-sample-all --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-latent --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 20 -k 1 --disable-gpu --disable-sample-all --disable-latent --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-sample-all -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-latent -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-sample-all --disable-latent -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-sample-all --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-latent --disable-iterative -alm ThompsonSampling

#python3 active_learning_main.py -m VAE-CF -r 50 -a 1 -l 0.0001 -i 300 -ali 100 -k 1 --disable-gpu --disable-sample-all --disable-latent --disable-iterative -alm ThompsonSampling
