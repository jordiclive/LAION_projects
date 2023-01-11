import argparse
import gc
import math
import os
import time
import fire
import torch
import pandas as pd

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--name", type=str, help="Name path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
    parser.add_argument("--ckpt_path", type=str, help="float16 or int8")

    return parser.parse_args()


def print_rank0(*msg,rank):
    if rank != 0:
        return
    print(*msg)

def main(ckpt_path='new_model.pt',model_name='EleutherAI/pythia-125m-deduped',val_path='val.json',kwargs = dict(
    device_map="balanced_low_0",
),batch_size=10,benchmark=None,max_source_length=512,target_length=150,hf_checkpoint=False,n_examples=21):
    t_start = time.time()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = torch.cuda.device_count()
    rank = local_rank



    print_rank0(f"Using {world_size} gpus",rank=rank)
    model_name = model_name
    print_rank0(f"Loading model {model_name}",rank=rank)
    dtype = torch.float16
    kwargs["torch_dtype"] = dtype

    if hf_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, **kwargs)
    else:
        tokenizer  = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.load_state_dict(torch.load(ckpt_path))

    val = pd.read_json(val_path)
    val = val.sample(n=n_examples,random_state=1)
    inputs = list(val['source'])
    inputs = ['<question> ' + i.strip() + ' <answer> ' for i in inputs]
    input_sentences = inputs
    if batch_size > len(input_sentences):
        # dynamically extend to support larger bs by repetition
        input_sentences *= math.ceil(batch_size / len(input_sentences))

    if benchmark:
        t_ready = time.time()

    ### Generate
    generate_kwargs = dict(max_new_tokens=target_length, do_sample=False)

    generate_kwargs = dict(use_cache=True,
                #decoder_start_token_id=model.decoder_start_token_id,
                num_beams=5,
                min_length=5,
                max_new_tokens=target_length,
                no_repeat_ngram_size = 3,
            )
    print_rank0(f"Generate args {generate_kwargs}",rank=rank)
    # inputs = input_sentences[: args.batch_size]

    def generate(inputs):
        """returns a list of zipped inputs, outputs and number of new tokens"""

        input_tokens = tokenizer.batch_encode_plus(inputs, max_length=max_source_length,
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt", )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to("cuda:0")

        outputs = model.generate(**input_tokens, num_beams=5, max_new_tokens=150, min_length=5)

        input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_tokens_lengths = [x.shape[0] for x in outputs]

        total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return inputs, outputs, total_new_tokens

    print_rank0("*** Running generate",rank=rank)
    t_generate_start = time.time()

    def chunks(xs, n):
        n = max(1, n)
        return (xs[i:i + n] for i in range(0, len(xs), n))
    inputs = list(chunks(inputs,n=batch_size))
    print(len(inputs[-1]))
    bs, cs, ds = [],[],[]
    for k in inputs:
        b, c, d = generate(k)
        bs.extend(b)
        cs.extend(c)
        ds.extend(d)

    df = pd.DataFrame({'source':bs,'prediction':cs,'target':list(val['target']), 'total_new_tokens':ds})
    df.to_csv(f'test_outputs_{ckpt_path}.csv',index=False)
    # t_generate_span = time.time() - t_generate_start
    # for i, o, _ in generated:
    #     print_rank0(f"{'-' * 60}\nin={i}\nout={o}\n",rank=rank)



if __name__ == '__main__':
    fire.Fire(main)












### Benchmark

# if args.benchmark:
#     # clear cache / free memory
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     print_rank0("*** Running benchmark")
#     # warm up
#     for i in range(1):
#         _ = generate()
#     torch.cuda.synchronize()
#
#     # benchmark
#     t0 = time.time()
#     cycles = 5
#     total_new_tokens_generated = 0
#     for i in range(cycles):
#         generated = generate()
#         total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)
#     torch.cuda.synchronize()
#     througput = (time.time() - t0) / (total_new_tokens_generated)
#     print_rank0(
#         f"""
# *** Performance stats:
# Throughput per token including tokenize: {througput*1000:.2f} msecs
# Start to ready to generate: {t_ready - t_start:.3f} secs
# Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
# Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
# """
   # )

