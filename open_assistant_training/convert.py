from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = "results/20221119_2105/epoch=0-step=20.ckpt"
output_path = "new_model.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)