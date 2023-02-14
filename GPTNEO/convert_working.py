from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
# Convert lightning/ds ckpt.

save_path = '/fsx/home-jordiclive/checkpoints/filter_epoch1_check'
ckpt_path = "model_large.pt"
model_name = 'google/flan-t5-xxl'
def convert_check(save_path,output_path,model_name,frozen=True,bfloat16=False,add_special=True):
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)

    # Rename weights
    ckpt_path = output_path
    state_dict = torch.load(ckpt_path)["state_dict"]
    print(state_dict.keys())
    for key in list(state_dict.keys()):
        if key.startswith("model."):
            state_dict[key[6:]] = state_dict.pop(key)

    if bfloat16 is False:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    if frozen:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        for n,p in model.named_parameters():
            if "embed" in n:
                state_dict[n] = model.state_dict()[n].to(dtype)

        torch.save(state_dict, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # todo if add special toks

    if not add_special:
        tokenizer.save_pretrained('hf_checkpoint')
        model.save_pretrained('hf_checkpoint', torch_dtype=dtype)
    else:

        tokenizer.add_special_tokens({"pad_token": "<|padding|>", "sep_token": "<|endoftext|>"})

        new_tokens = ['<sp_token_answer>']
        new_tokens_vocab = {}
        new_tokens_vocab["additional_special_tokens"] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab["additional_special_tokens"].append(t)
        num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)

        tokenizer.save_pretrained('hf_checkpoint')
        model.save_pretrained('hf_checkpoint', torch_dtype=dtype)
        model.resize_token_embeddings(len(tokenizer))

