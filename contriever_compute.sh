#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g40423"
#SBATCH --job-name=contriever
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_flan/bin/activate
cd /admin/home-jordiclive/LAION_projects/summarization_data_prep/processing/contriever
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0"
python contriever_code.py