from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
# Convert lightning/ds ckpt.

save_path = '/fsx/home-jordiclive/checkpoints/filter_epoch1_check'
output_path = "model_large.pt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)


output_path = "new_model_large.pt"

# Rename weights
ckpt_path = 'model_large.pt'
state_dict = torch.load(ckpt_path)["state_dict"]
print(state_dict.keys())
for key in list(state_dict.keys()):
    if key.startswith("model."):
        state_dict[key[6:]] = state_dict.pop(key)


# If freeze embeddings add back in.
model_name = 'google/flan-t5-xxl'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
for key in ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
    state_dict[key] = model.state_dict()[key].to(state_dict['lm_head.weight'].dtype)

# save new model
torch.save(state_dict, "new_model_large.pt")

# Save as HF model.
ckpt_path = "new_model_large.pt"
model.load_state_dict(torch.load(ckpt_path))
model.save_pretrained('hf_checkpoint',torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('hf_checkpoint')


# # save_path = '/admin/home-jordiclive/LAION_projects/FLAN_code/results/20230128_2007/epoch=0-step=1.ckpt'
# save_path = '/admin/home-jordiclive/LAION_projects/FLAN_code/results/20230128_2210/epoch=0-step=1.ckpt'
# # # # save_path = "/admin/home-jordiclive/LAION_projects/open_assistant_training/results/20230111_1553/epoch=0-step=24.ckpt"
# output_path = "model_large.pt"
# convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)



# ckpt_path = 'new_model_large.pt'
# state_dict = torch.load(ckpt_path)["state_dict"]
# for key in list(state_dict.keys()):
#     if key.startswith("model."):
#         state_dict[key[6:]] = state_dict.pop(key)
# torch.save(state_dict, "new_model_large.pt")
# model_name = 'EleutherAI/pythia-13b-deduped'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# QA_SPECIAL_TOKENS = {"Question": "<question>", "Answer": "<answer>"}
# new_tokens = ['<question>', '<answer>']
# new_tokens_vocab = {}
# new_tokens_vocab["additional_special_tokens"] = []
# for idx, t in enumerate(new_tokens):
#     new_tokens_vocab["additional_special_tokens"].append(t)
# num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)

# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# model.resize_token_embeddings(len(tokenizer))

# model.load_state_dict(torch.load(ckpt_path))

# model.save_pretrained('hf_checkpoint',torch_dtype=torch.float16)
# tokenizer.save_pretrained('hf_checkpoint')

