from transformers import GPTNeoXForCausalLM, GPTNeoForCausalLM, AutoTokenizer
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("model")
#args = parser.parse_args()
model = "/admin/home-jordiclive/LAION_projects/open_assistant_training/hf_checkpoint/"

model = GPTNeoXForCausalLM.from_pretrained(model).half().to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model)


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