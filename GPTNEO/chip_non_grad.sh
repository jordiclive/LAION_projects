#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80n140"
#SBATCH --job-name=openassistant
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_deep/bin/activate
cd /admin/home-jordiclive/LAION_projects/GPTNEO
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
srun python train.py --precision "16" --grad_checkpoint False --lr_flat False --warmup_steps 1000 --train_batch_size 8 --eval_batch_size 8 --wb_name "without"