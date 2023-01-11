from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

save_path = '/fsx/home-jordiclive/open_assistant_training/results/20230111_1503/epoch=0-step=40.ckpt'
# save_path = "/admin/home-jordiclive/LAION_projects/open_assistant_training/results/20230111_1553/epoch=0-step=24.ckpt"
output_path = "new_model_large.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

import torch
ckpt_path = 'new_model_large.pt'
state_dict = torch.load(ckpt_path)["state_dict"]
for key in list(state_dict.keys()):
    if key.startswith("model."):
        state_dict[key[6:]] = state_dict.pop(key)
torch.save(state_dict, "new_model_large.pt")