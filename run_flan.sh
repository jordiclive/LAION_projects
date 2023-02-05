#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80n140"
#SBATCH --job-name=flan
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_flan/bin/activate
cd /admin/home-jordiclive/LAION_projects/FLAN_code
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
srun python train.py --train_batch_size 4 --val_check_interval 0.12 --limit_val_batches 0.5 --wb_name "bs4-retrain-more-prompts-13B" --gradient_accumulation_steps 2 --learning_rate 0.0003 --data_path '/admin/home-jordiclive/LAION_projects/summarization_data_prep/processing/final_dataset' --resume_from_checkpoint '/fsx/home-jordiclive/checkpoints/20230204_1242/epoch=0-step=26046.ckpt/checkpoint'