#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80n60"
#SBATCH --job-name=flan
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_flan/bin/activate
cd /admin/home-jordiclive/LAION_projects/FLAN_code
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
srun python train.py

