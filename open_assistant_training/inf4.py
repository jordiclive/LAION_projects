from transformers import AutoModelForCausalLM, GPTNeoForCausalLM, AutoTokenizer
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("model")
#args = parser.parse_args()
model = "hf_checkpoint/"

model = AutoModelForCausalLM.from_pretrained(model).half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-13b-deduped')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
QA_SPECIAL_TOKENS = {"Question": "<question>", "Answer": "<answer>"}
new_tokens = ['<question>', '<answer>']
new_tokens_vocab = {}
new_tokens_vocab["additional_special_tokens"] = []
for idx, t in enumerate(new_tokens):
    new_tokens_vocab["additional_special_tokens"].append(t)
num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)


while True:
    text = input("\n\nInput text to prompt the model: ")
    text = str(text)
    if len(text) == 0:
        continue
    ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    # add the length of the prompt tokens to match with the mesh-tf generation
    max_length = 400 + ids.shape[1]

    gen_tokens = model.generate(
        ids,
        do_sample=True,
        min_length=max_length,
        max_length=max_length,
        temperature=0.9,
        use_cache=True
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print("Text generated:")
    print(gen_text)
# # text = input("\n\nInput text to prompt the model: ")
# # text = str(text)
# text = "<question>What is a good way to train a large language model with multiple gpus?"
# ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
# # add the length of the prompt tokens to match with the mesh-tf generation
# max_length = 400 + ids.shape[1]
#
# gen_tokens = model.generate(
#     ids,
#     do_sample=True,
#     min_length=max_length,
#     max_length=max_length,
#     temperature=0.9,
#     use_cache=True
# )
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print("Text generated:")
# print(gen_text)