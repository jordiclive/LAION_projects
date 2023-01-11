from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = "/admin/home-jordiclive/LAION_projects/open_assistant_training/results/20230111_1553/epoch=0-step=24.ckpt"
output_path = "new_model.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)