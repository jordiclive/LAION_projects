#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80n140"
#SBATCH --job-name=filtersummarization
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out



source /admin/home-jordiclive/jordan_flan/bin/activate
cd /admin/home-jordiclive/LAION_projects/FLAN_code
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
srun python train.py --train_batch_size 4 --val_check_interval 0.5 --wb_name "13B-bs4-filter-0.7_restart" --gradient_accumulation_steps 2 --learning_rate 0.0003 --data_path '/admin/home-jordiclive/LAION_projects/summarization_data_prep/processing/new_prompts/final_dataset/0.7_final_dataset' --resume_from_checkpoint "/fsx/home-jordiclive/checkpoints/20230206_1002/epoch=1-step=48825.ckpt/"