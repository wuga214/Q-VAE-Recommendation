sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_autorec_parameter_tuning.sh -o log/log_autorec.txt -e log/log_error_autorec.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_bpr_parameter_tuning.sh -o log/log_bpr.txt -e log/log_error_bpr.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_cdae_parameter_tuning.sh -o log/log_cdae.txt -e log/log_error_cdae.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_cml_parameter_tuning.sh -o log/log_cml.txt -e log/log_error_cml.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_ifvae_parameter_tuning.sh -o log/log_ifvae.txt -e log/log_error_ifvae.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_vae_parameter_tuning.sh -o log/log_vae.txt -e log/log_error_vae.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_wrmf_parameter_tuning.sh -o log/log_wrmf.txt -e log/log_error_wrmf.txt
sbatch --nodes=1 --time=24:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_puresvd_parameter_tuning.sh -o log/log_puresvd.txt -e log/log_error_puresvd.txt
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_plrec_parameter_tuning.sh -o log/log_plrec.txt -e log/log_error_plrec.txt
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus=4 --gres=gpu:1 movielens_20m_nceplrec_parameter_tuning.sh -o log/log_nceplrec.txt -e log/log_error_nceplrec.txt


