
import gc
import math
import os
import time
from argparse import ArgumentParser
import argparse
import gc
import math
import os
import time
import fire
import torch
import pandas as pd
import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig


t_start = time.time()

num_tokens = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"



local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


### Model loading and instantiating on GPU (via ZeRO)
ckpt_path='new_model_large.pt'
model_name='EleutherAI/pythia-13b-deduped'
val_path='val.json'
hf_checkpoint = False
n_examples = 20
dtype = torch.float16
kwargs = dict(

)
kwargs["torch_dtype"] = dtype

if hf_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, **kwargs)
else:
    tokenizer = AutoTokenizer.from_pretrained("checkpoint-curr-best_20230111_1557")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(ckpt_path))

val = pd.read_json(val_path, orient='split')
val = val.sample(n=n_examples, random_state=1)
inputs = list(val['source'])
inputs = ['<question> ' + i.strip() + '<answer>' for i in inputs]
input_sentences = inputs

print_rank0(f"*** Loading the model {model_name}")



# XXX: can't automatically derive dtype via config's `from_pretrained`
dtype =  torch.float16

model_hidden_size = 5120
train_batch_size = 1 * world_size

ds_config = {
    "fp16": {
        "enabled": dtype == torch.float16,
    },
    "bf16": {
        "enabled": dtype == torch.bfloat16,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 0,
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}

# if args.cpu_offload and args.nvme_offload_path:
#     raise ValueError("Use one of --cpu_offload or --nvme_offload_path and not both")
#
# if args.cpu_offload:
#     ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)
#
# if args.nvme_offload_path:
#     ds_config["zero_optimization"]["offload_param"] = dict(
#         device="nvme",
#         pin_memory=True,
#         nvme_path=args.nvme_offload_path,
#         buffer_size=4e9,
#     )

dschf = HfDeepSpeedConfig(ds_config)  # this tells from_pretrained to instantiate directly on gpus


torch.cuda.empty_cache()
gc.collect()
deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)


# if args.benchmark:
#     deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()

print_rank0(ds_config)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
model = ds_engine.module

# if args.benchmark:
#     t_ready = time.time()
#     deepspeed.runtime.utils.see_memory_usage("start-of-generate", force=True)
#

### Generate

print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]
batch_size = 1
if batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

print_rank0(f"Generate args {generate_kwargs}")
inputs = input_sentences[: batch_size]


def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


# XXX: this is currently doing world_size streams on world_size gpus, so we can feed it different inputs on each! and hence the time can be divided by world_size

print_rank0("*** Running generate")
t_generate_start = time.time()
pairs = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in pairs:
    print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")


### Benchmark


    # clear cache / free memory
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("end-of-generate", force=True)

    print_rank0("*** Running benchmark")

    # warm up
    for i in range(1):
        _ = generate()
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate()
        total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)

    torch.cuda.synchronize()
    # note that we actually generate world_size unique streams (though the benchmark feeds the same inputs)
    total_new_tokens_generated *= world_size
    througput = (time.time() - t0) / (total_new_tokens_generated)
    print_rank0(
        f"""
*** Performance stats:
Throughput per token including tokenize: {througput*1000:.2f} msecs
"""
    )