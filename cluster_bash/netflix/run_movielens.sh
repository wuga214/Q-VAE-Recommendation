#!/usr/bin/env bash
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_autorec_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_bpr_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_cdae_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_cml_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_ifvae_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_vae_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_wrmf_parameter_tuning_part1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_wrmf_parameter_tuning_part2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_wrmf_parameter_tuning_part3.sh
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_puresvd_parameter_tuning.sh
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_plrec_parameter_tuning.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 netflix_nceplrec_parameter_tuning.sh
