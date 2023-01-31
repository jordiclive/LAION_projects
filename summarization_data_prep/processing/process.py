import glob
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import glob
import glob
import re

import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import random

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")



def word_count(x):
    count = len(re.findall(r"\w+", x))
    return count


def token_count(x):
    return len(tokenizer.encode(x))


def print_quantiles(x):
    for i in [0.1, 0.5, 0.9]:
        print(i, x.quantile(i))
    # for i in [0.5,0.6,0.9, 0.925, 0.95, 0.975, 0.99]:
    #     print(i, x.quantile(i))


# path = '/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/scored_summarization_datasets/datasets/*'
# for k in glob.glob(path):
#     if 'train' in k:
#         df = pd.read_parquet(k)
#         df = df.sample(1000)
#         print(f"""\n\n---------------------\n\n{k.split('/')[-1]}""")
#         # print('\n\ntext Word Count')
#         # print_quantiles(df['text'].apply(word_count))
#         # print('\n\nSummary Word Count')
#         # print_quantiles(df['summary'].apply(word_count))
#         print('\n\ntext Token Count')
#         print_quantiles(df['text'].apply(token_count))
#         print('\n\nSummary Token Count')
#         print_quantiles(df['summary'].apply(token_count))
#


prompt_dict = {
    "scitldr": {
        "main": [
            "Given the following scientific article, provide a TL;DR summary:",
            "Summarize the following article in {r1}-{r2} words:",
            "Write a summary of the following text that has {r1}-{r2} words:",
            "Summarize the following scientific article in {r1}-{r2} words in a TL;DR format:",
        ],
        "pruned_candidates": [
            "Give an overview of the scientific article provided here:",
            "Write a summary of the following text that contains {r1}-{r2} words",
            "Given the following scientific article give a TL;DR summary:",
            "Write a summary of the following text in {r1} to {r2} words:",
            "Describe the following article in {r1}-{r2} words:",
            "Please summarize the following scientific article in {r1} to {r2} words in a TL;DR",
            "Create a summary of the following text with {r1} to {r2} words:",
            "Make the following scientific article in {r1}-{r2} words in a TL",
            "summarize the following article in {r1} to {r2} words:",
            "Write a summary of the following text which is between {r1} and {r2} words:",
            "Give a scientific summary of the following article:",
            "Write a summary of the following text that contains 10-35 words:",
            "Summarise the following scientific article in {r1}-{r2} words in a TL;DR format:",
            "Summarize the following scientific article in {r1}-{r2} words in TL'DR format:",
            "Give a TL;DR about the following scientific article:",
            "Briefly summarize the following scientific article in {r1}-{r2} words in a TL;DR format:",
            "Given the following scientific article, provide a brief TL;DR summary:",
            "Write a summary of the following text of {r1}-{r2} words:",
            "Summarize the following scientific article in {r1}-{r2} words in a TL DR format",
            "Make a complete summary of the following scientific article in {r1}-{r2} words in a TL DR",
            "Give a quick summary of the following scientific articles:",
            "The following scientific article should be summarized in {r1}-{r2} words.",
            "Give a summary of the scientific article.",
            "Summarize the following scientific article in {r1}-{r2} words in a TL;DR format",
            "Write a summary of the following text with {r1}-{r2} words:",
            "The following article should be summarized in {r1}-{r2} words",
            "Given the following scientific article provide a short summary:",
            "Generally summarize the following article in {r1}-{r2} words:",
            "When presenting the following scientific article provide a TL;DR summary:",
            "Give a summary of the scientific article",
            "Summarize the following scientific article in {r1}-{r2} words in TL;DR format:",
            "What should be the scientific summary of the following article?",
            "Write a summary of the text.",
            "summarize the following article in {r1}-{r2} words:",
            "Briefly summarize the following scientific article in in a TL;DR format",
            "Write a summary of the following text of {r1} to {r2} words:",
            "Write a summary of the following text that is {r1}-{r2} words:",
            "Write a summary of the following text that is {r1}-{r2} words long:",
            "Given the following scientific article provide a brief summary:",
            "Summarize the following scientific article in {r1}-{r2} words in a TL;DR format:",
            "Write a summary of the following text that has {r1}-{r2} words:",
            "Write a summary of the following text which is {r1}-{r2} words:",
            "Write a summary of the following text which has {r1}-{r2} words:",
            "Summarize the following article in {r1}-{r2} words:",
        ],
    },
    "wikihow": {
        "main": [
            "Produce an article summary including outlines of each paragraph of the following article:",
            "Summarize the following article, by including an outline of each paragraph:",
            "Write a summary of the following article, including paragraph outlines that has {r1}-{r2} words:",
            "Summarize the following article in {r1}-{r2} words:",
        ],
        "pruned_candidates": [
            "Produce an article summary including the outline of each paragraph of the following article:",
            "Write the following article as a summary by including an outline of each paragraph:",
            "Please summarize the following article by including an outline of each paragraph:",
            "Write a {r1}-{r2} word summary of the following article, including paragraph outlines:",
            "Write a {r1}-{r2} word summary of the following article including paragraph outlines:",
            "Briefly summarize the following article in {r1}-{r2} words:",
            "Make an article summary including the outline of every paragraph of the following article:",
            "Write a {r1}-{r2} words summary of the following article including paragraphs outlines:",
            "Write a summary of the following article including paragraph outlines that has {r1}-{r2} words:",
            "Summarize the following article by including an outline of each paragraph:",
            "Produce a summary of the article (with outlines of each paragraph) of the following article:",
            "You can summarise the following article in {r1}-{r2} words:",
            "summarize the following article by including an outline of each paragraph:",
            "Summarize the following article by adding an outline of each paragraph:",
            "Give a summary of the following article, including paragraph outlines, that have {r1}-{r2} words:",
            "Create an article summary including each paragraph of the following article:",
            "Write a summary of the following article, including paragraph outlines that, is {r1}-{r2} words",
            "Write a summary of the following article, including the outline of paragraphs, that contains {r1}-{r2} words:",
            "Summarize the following article in words {r1}-{r2} words",
            "Write a summary of {r1}-{r2} words of the following article including paragraph descriptions:",
            "An outline of each paragraph is required to summarize the following article:",
            "Can you summarize the following article in {r1}-{r2} words for me:",
            "Give the following WikiHow article a summary:",
            "Write a summary of the following article, including paragraph outline that has {r1}-{r2} words:",
            "Write a summary of the following article, including paragraphs outlines:",
            "Summarize the following article by including a summarized outline of each paragraph:",
            "Produce a brief article with details of each paragraph of the following article:",
            "Write a summary of the following article including the paragraph outlines:",
            "Summarise the following article in {r1}-{r2} words:",
            "Summarize the following article by introducing an outline of each paragraph: ",
            "Create an article summary including the outline of each paragraph of the following article: ",
            "Synthesize the following article by including an outline of each paragraph:",
            "Make an article summary including an outline for each paragraph of the following article:",
            "produce a summary article e.g. outline each paragraph of the following article:",
            "Please summarize the following article in {r1}-{r2} words",
            "Produce a summary of the article including the outline of each paragraph of the following article:",
            "Write a summary of the following article including paragraph outlines so the summary has {r1}-{r2} words",
            "Produce an article summary including an outline of each paragraph of the following article:",
            "Produce an article summary including outlines for each paragraph of the following article:",
            "The following article should be summarized in {r1}-{r2} words:",
            "summarize the following article by including the outline of each paragraph :",
            "I like to summarize the following article in {r1}-{r2} words:",
            "Summarize the following article in {r1}-{r2} words:",
        ],
    },
    "newsroom": {
        "main": [
            "Produce an article summary of the following news article:",
            "Given the following article, provide a summary:",
            "Summarize the following article in {r1}-{r2} words:",
            "Write a {r1}-{r2} word summary of the following text:",
        ],
        "pruned_candidates": [
            "Produce a summary article of the following news:",
            "Give a brief summary of the following article:",
            "Write a summary of the following text that has {r1} to {r2} words:",
            "Describe the following article in {r1}-{r2} words:",
            "Produce a summary of the following news article:",
            "The text has {r1}-{r2} words:",
            "Write a summary of the text:",
            "Write a summary of the following text of {r1}-{r2} words:",
            "Produce an article summary of the following news article:",
            "Write a summary of the following text with {r1}-{r2} words.",
            "Provide a summary of the following news article:",
            "Creating an article summary of the following news article:",
            "Give a summary of this article:",
            "Write a summary of the following text with {r1} to {r2} words:",
            "Give a summary of the article:",
            "Give a summary of the following article in {r1}/{r2} words:",
            "Write a summary of the following text in {r1}-{r2} words:",
            "Tell me the summary of this article.",
            "Please summarize the following article in {r1}-{r2} words:",
            "Write a summary of the following text that contains {r1}-{r2} words:",
            "Summarise the following article in {r1} to {r2} words:",
            "Give a summary to the following article:",
            "Make a summary of the following article in {r1}-{r2} words:",
            "Give the following article a summary:",
            "Summary of the following article in {r1}-{r2} words:",
            "Write a summary of the following text with {r1}-{r2} words:",
            "produce an article summary of the following news article:",
            "Can you summarize the following article?",
            "summarize the following article in {r1}-{r2} words:",
            "Give a summary of the following article:",
            "Write a summary of the following text that is {r1}-{r2} words:",
            "Given the following article give a summary:",
            "Give the following article a brief summary:",
            "Given the following article provide a summary:",
            "Do you want to summarize the following article in {r1}-{r2} words:",
            "Write a summary of the following text that has {r1}-{r2} words:...",
            "Create a summary of the following news article:",
            "The following article should be summarized in {r1}-{r2} words:",
            "Write an article summary of the following news article:",
            "Summarize the following article in {r1}-{r2} words:",
            "Describe the following news story briefly:",
        ],
    },
    "samsum": {
        "main": [
            "Briefly summarize in third person the following conversation:",
            "Summarize the following conversation in {r1}-{r2} words:",
            "Give a third person summary of the following conversation:",
            "Summarize the following in {r1}-{r2} words:",
        ],
        "pruned_candidates": [
            "Summarize the following words in {r1} to {r2} words:",
            "Give a third person summary of the following conversation:",
            "Summarize the following conversation in {r1}-{r2} words:",
            "Tell me the following conversation in third person:",
            "Summarize the following conversation in {r1} to {r2} words:",
            "Summarize the following in {r1} to {r2} words:",
            "In the third person summarize the following conversation:",
            "Can you explain in third person the following conversation:",
            "Summarize the following in {r1}-{r2} words: ",
            "Briefly summarize the following conversation in the third person:",
            "The following conversation should be summarized in {r1} or {r2} words.",
            "Summarize the following in {r1}-{r2} words:",
            "I'm going to summarise in the third person the following conversation:",
            "Write the following in {r1}-{r2} words:",
            "Recap the following conversation briefly in the third person:",
            "Give a third person a summary of the following conversation:",
            "Summarize the following information in {r1}-{r2} words:",
            "Please summarize the conversation:",
            "Can you also summarize the conversation in third person?",
            "Give a summary of the following conversation in third person:",
            "summarize the following in a mere {r1}-{r2} words:",
            "The following conversation should be summarized in {r1}-{r2} words:",
            "Can you summarize the following conversation in {r1}-{r2} words:",
            "Give a summary in the third person of the following conversation:",
            "Please briefly summarize in third person the following conversation:",
            "briefly summarize in third person the following conversation:",
            "Give a summary of the conversation to a third person",
            "Synthesize the following conversation in {r1}-{r2} words:",
            "Speak in the third person briefly about the following conversation:",
            "give a third person summary of the following conversation:",
            "Write up the main points of the following conversation:",
            "In {r1}-{r2} words, summarize the following:",
            "Summarize the following in {r1}/{r2} words:",
            "summarize the following conversation in {r1}-{r2} words:",
            "We should quickly summarize in third person the following conversation:",
            "Please summarise the following in {r1}-{r2} words:",
            "Describe in third person the following conversation:",
            "The following conversation should be summarized in third person:",
        ],
    },
    "xsum": {
        "main": [
            "Given the following news article, summarize the article in one sentence:",
            "Produce a single sentence summary of the following:",
            "Summarize the following article in {r1}-{r2} words:",
            "Write a {r1}-{r2} word summary of the following text:",
        ],
        "pruned_candidates": [
            "How would you summarize a news article in a single sentence?",
            "Let us think of the following news article in a sentence:",
            "Given the following article, give your summary of the article.",
            "give a summary of the following in one sentence:",
            "Write a single sentence summary of the following:",
            "Give the following news article a succinct sentence:",
            "Write a short summary of the following text:",
            "Write a summary of the following text in {r1} to {r2} words:",
            "Write a summary of the following text in {r1}-{r2} words",
            "A single sentence summary of the following is required:",
            "Give me the following news article in one sentence:",
            "A single sentence summary is required:",
            "Make a brief summary of the following article in {r1} to {r2} words:",
            "write a {r1}-{r2} word summary of the following text:",
            "Summarize the following article in {r1} {r2} words:",
            "Adapt the following article in {r1}/{r2} words:",
            "The article should be summarized in one sentence:",
            "Summarize the following article in {r1} - {r2} words:",
            "Write a {r1}-{r2} word summary of the following text:",
            "Write a {r1}-{r2}-word summary of the following text:",
            "Produce a single sentence summary of the following: ",
            "Give a summary of the following in one sentence:",
            "Given the following news article summarize the article in one sentence:",
            "The following article should be summarized in {r1}-{r2} words",
            "Show a complete summary of the following article in {r1} - {r2} words:",
            "Describe to me the following article in one sentence:",
            "Produce a single sentence summary of the following:",
            "Write a summary of the text.",
            "Given the following news article, summarize it in one sentence:",
            "Provide a single sentence summary of the following:",
            "summarize the following paragraphs in {r1}-{r2} words:",
            "Write a description of the article in {r1}-{r2} words:",
            "Explain the following article in {r1}-{r2} words:",
            "This article should be summarized in one sentence: ",
            "Give the following news article in one sentence:",
            "Produce a single sentence summary of:",
            "Summarise the following article in {r1}-{r2} words:",
            "Prepare a summary sentence of the following:",
            "Can you summarize the following article in {r1}-{r2} words:",
            "Recap the following news article one sentence:",
            "Send a single sentence summary of the following:",
            "Given the following news article, summarize the article in one sentence:",
            "Create a single sentence summary of the following:",
            "Summarize the following article in {r1}-{r2} words:",
        ],
    },
    "cnn_dailymail": {
        "main": [
            "Produce an article summary of the following news article:",
            "Given the following article, provide a summary:",
            "Summarize the following article in {r1}-{r2} words:",
            "Write a {r1}-{r2} summary of the following text:",
        ],
        "pruned_candidates": [
            "Produce a summary article of the following news:",
            "Give a brief summary of the following article:",
            "Write a summary of the following text that has {r1} to {r2} words:",
            "Describe the following article in {r1}-{r2} words:",
            "Produce a summary of the following news article:",
            "The text has {r1}-{r2} words:",
            "Write a summary of the text:",
            "Write a summary of the following text of {r1}-{r2} words:",
            "Produce an article summary of the following news article:",
            "Write a summary of the following text with {r1}-{r2} words.",
            "Provide a summary of the following news article:",
            "Creating an article summary of the following news article:",
            "Give a summary of this article:",
            "Write a summary of the following text with {r1} to {r2} words:",
            "Give a summary of the article:",
            "Give a summary of the following article in {r1}/{r2} words:",
            "Write a summary of the following text in {r1}-{r2} words:",
            "Tell me the summary of this article.",
            "Please summarize the following article in {r1}-{r2} words:",
            "Write a summary of the following text that contains {r1}-{r2} words:",
            "Summarise the following article in {r1} to {r2} words:",
            "Give a summary to the following article:",
            "Make a summation of the following article in {r1}-{r2} words:",
            "Give the following article a summary:",
            "Summary of the following article in {r1}-{r2} words:",
            "Write a summary of the following text with {r1}-{r2} words:",
            "produce an article summary of the following news article:",
            "Can you summarize the following article?",
            "summarize the following article in {r1}-{r2} words:",
            "Give a summary of the following article:",
            "Write a summary of the following text that is {r1}-{r2} words:",
            "Given the following article give a summary:",
            "Give the following article a brief summary:",
            "Given the following article provide a summary:",
            "Do you want to summarize the following article in {r1}-{r2} words:",
            "Write a summary of the following text that has {r1}-{r2} words:...",
            "Create a summary of the following news article:",
            "The following article should be summarized in {r1}-{r2} words:",
            "Write an article summary of the following news article:",
            "Summarize the following article in {r1}-{r2} words:",
            "Describe the following news story briefly:",
        ],
    },
    "tldr-challenge": {
        "main": [
            "Produce a short summary of the following social media post:",
            "Provide a very short, TL;DR summary of the following post:",
            "Summarise in {r1}-{r2} words the following post:",
            "Give a TL;DR Summary of the following using {r1}-{r2} words:",
        ],
        "pruned_candidates": [
            "The following post has a TL;DR summary. What is it?",
            "Give a TL;DR summary of the following using {r1}-{r2} words: '",
            "Honestly, give a very brief summary of the following post:",
            "Give a very short and TL;DR summary of the following blog post:",
            "Make a short summary of the following social media post:",
            "Give a TL;DR summary of the following in {r1}-{r2} words:",
            "Summarise in {r1}-{r2} words the following post:",
            "Give a TL;DR summary of the following using {r1}-{r2} words:",
            "Give a summary of the following using {r1}-{r2} words:",
            "A short summary of a social media post is required.",
            "Tell me the following post in {r1}-{r2} words:",
            "Provide a very short summary of the following post:",
            "Give a very short summary of the following post:",
            "Give me a very short summary of the following post. TL DR",
            "Summarize in {r1}-{r2} words the following post:",
            "Give a very short summary of the following post:'",
            "Give a TL;DR, summarizing the following using {r1}-{r2} words:",
            "give a very short, TL;DR summary of the following post:",
            "Give a TL;DR summary of the following using {r1}/{r2} words:",
            "Give a very short summary of the following post TL;DR:",
            "Summarise the following post in {r1} to {r2} words:",
            "Produce a short summary of the following social media post:",
            "give a summary of the following using {r1}-{r2} words:",
            "Give a brief summary of the following using {r1} to {r2} words:",
            "Give a short summary of the following post:",
            "Basically sum up in {r1}-{r2} words the following post:",
            "Summarise the following Post in {r1} or {r2} words:",
            "Give a summary of the following in {r1}/{r2} words:",
            "Put the following post in {r1}-{r2} words:",
            "Tell me the following paragraph in {r1}-{r2} words:",
            "offer a very short overview of the following post:",
            "If you had to summarize the following post in {r1}-{r2} words:",
            "make a short summary of the following social media post:",
            "Give a TL;DR Summary of the following in {r1} to {r2} words:",
            "Give a very brief summary of the following article:",
            "Write a short summary of the following social media post:",
            "Create a short version of the following social media post:",
            "provide a very short summary of the following post:",
            "Write a quick summary of the following social media post:",
            "Create a short overview of the following social media post:",
            "Provide a very short, TL;DR summary of the following post:",
            "Please give a summary of the following in {r1} to {r2} words:",
            "Provide a very short summary of the post:",
            "Give a tl;dr Summary of the following using {r1}-{r2} words:",
            "Summarise the following post in {r1}-{r2} words:",
        ],
    },
    "wikipedia-summary": {
        "main": [
            "Summarize the following in {r1}-{r2} words:",
            "Write a {r1}-{r2} word summary of the following text:",
            "Summarize in {r1}-{r2} words:",
            "Provide a summary of the following article:",
        ],
        "pruned_candidates": [
            "Describe the following article: ",
            "Summarise in {r1} to {r2} words:",
            "summarize the following in {r1}-{r2} words:",
            "Give a brief summary of the following text:",
            "Provide a summary of the article:",
            "Summarize the following in {r1} to {r2} words:",
            "summarize the following in {r1}{r2} words:",
            "How can I summarize in {r1}-{r2} words?",
            "Provide a summary of the article",
            "In {r1}-{r2} words, summarize the following.",
            "Summarize in {r1}-{r2} words",
            "Summarize the following in {r1}-{r1} words:",
            "Summarize the following in {r1} - {r2} words:",
            "Can you provide a summary of the following article:",
            "Summarise in {r1}-{r2} words:",
            "Give a summary of the following article:",
            "Provide a summary of the following article:",
            "Summarize the following in {r1}-{r2} words:",
            "Can you summarise the following in {r1}-{r2} words:",
            "Summarize in {r1}-{r2} words:",
            "Write a {r1}-{r2} word summary of the following text:",
            "Please provide a summary of the following article:",
            "Write a summary of the following text:",
            "write a {r1}-{r2} word summary of the following text:",
            "Please summarize the following in {r1} to {r2} words:",
            "Give a summary ({r1}-{r2} words) of the following:",
            "give a summary of the following article:",
            "In {r1}-{r2} words, summarize.",
            "Write a summary of the text.",
            "show a summary of the following article:",
            "Summarize in {r1} to {r2} words:",
            "Summary in {r1}-{r2} words:",
            "Write a {r1}-{r2} summary of the following text:",
        ],
    },
}


# xsum_prompts = [
#     "Given the following news article, summarize the article in one sentence:"
# ]
# cnn_dailymail_prompts = ["Produce an article summary of the following news article:"]


# Paraphrase Models
# geckos/pegasus-fined-tuned-on-paraphrase
# prithivida/parrot_paraphraser_on_T5

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
#
# def generate(text, model="prithivida/parrot_paraphraser_on_T5"):
#     tokenizer = AutoTokenizer.from_pretrained(model)
#
#     model = AutoModelForSeq2SeqLM.from_pretrained(model)
#
#     text = f"paraphrase: {text} </s>"
#     enc = tokenizer(text, return_tensors="pt")
#     tokens = model.generate(
#         **enc,
#         num_return_sequences=3,
#         output_scores=True,
#         do_sample=True,
#         top_k=120,
#         top_p=0.95,
#     )
#     res = []
#     for output in tokens:
#         line = tokenizer.decode(
#             output, skip_special_tokens=True, clean_up_tokenization_spaces=True
#         )
#         res.append(line)
#     return res
#
#
# model_name = "geckos/pegasus-fined-tuned-on-paraphrase"
# torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
#
#
# def get_response(input_text, num_return_sequences):
#     batch = tokenizer.prepare_seq2seq_batch(
#         [input_text],
#         truncation=True,
#         padding="longest",
#         max_length=60,
#         return_tensors="pt",
#     ).to(torch_device)
#     translated = model.generate(
#         **batch,
#         num_return_sequences=num_return_sequences,
#         output_scores=True,
#         do_sample=True,
#         top_k=120,
#         top_p=0.95,
#         temperature=0.5,
#     )
#     tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#     return tgt_text
#
#
# from sentence_splitter import SentenceSplitter, split_text_into_sentences
#
# splitter = SentenceSplitter(language="en")
#
#
# def paraphraze(text):
#     sentence_list = splitter.split(text)
#     paraphrase = []
#     # print(sentence_list)
#
#     a = get_response(text, 5)
#     # print(a)
#     return a
#
#
# def get_prompt(prompt_dict, dataset_name):
#     for key, value in prompt_dict.items():
#         if key in dataset_name:
#             return value["main"]
#
#
# def generate_more_prompts(prompts):
#     prompt_candidates = []
#     for prompt in prompts:
#         for i in range(5):
#             prompt_candidates.extend(generate(prompt))
#         prompt_candidates.extend(paraphraze(prompt))
#     prompt_candidates = list(set(prompt_candidates))
#     return prompt_candidates
#
#
# ## Calculate Prompts
# path = "../scored_summarization_datasets/datasets/*"
# for k in glob.glob(path):
#     k = "wikipedia-summary"
#     l = [
#         "scitldr",
#         "wikihow",
#         "newsroom",
#         "multixscience",
#         "samsum",
#         "cnn_dailymail",
#         "billsum",
#         "xsum",
#     ]
#
#     # if "train" in k and not (any(x in k for x in l)):
#     if not (any(x in k for x in l)):
#
#         # df = pd.read_parquet(k)
#         # print(k.split("/")[-1])
#         # df = df.sample(10000, replace=True)
#         # print("Dataset Length:", len(df))
#         # print(f"""\n\n---------------------\n\n{k.split('/')[-1]}""")
#         # print("\n\ntext Word Count")
#         # print_quantiles(df["text"].apply(word_count))
#         # print("\n\nSummary Word Count")
#         # print_quantiles(df["summary"].apply(word_count))
#         # # print('\n\ntext Token Count')
#         # # print_quantiles(df['text'].apply(token_count))
#         # # print('\n\nSummary Token Count')
#         # # print_quantiles(df['summary'].apply(token_count))
#         #
#         # print("\n\ntext token calc Count")
#         # print_quantiles(1.42 * df["text"].apply(word_count))
#         # print("\n\nSummary token calc Count")
#         # print_quantiles(1.42 * df["summary"].apply(word_count))
#         # print("\n\ntext Token Count")
#
#         prompts = get_prompt(prompt_dict, k.split("/")[-1])
#         prompts2 = []
#         for s in prompts:
#             s = s.replace("{r1}", "10")
#             s = s.replace("{r2}", "35")
#             prompts2.append(s)
#         prompt_candidates = generate_more_prompts(prompts2)
#         prompt_candidates2 = []
#         for l in prompt_candidates:
#             l = l.replace("10", "{r1}")
#             l = l.replace("35", "{r2}")
#             prompt_candidates2.append(l)
#         print(prompt_candidates2)
#         break
# Process wikisummary

import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import re

def word_count(x):
    count = len(re.findall(r"\w+", x))
    return count




def myround(x, base=5):
    return base * round(x / base)


def get_prompted_comment(x, prompts, q1, q_mean, q2):
    no = random.random()
    if no < 0.6:
        prompt = random.choice(prompts["main"])
    else:
        prompt = random.choice(prompts["pruned_candidates"])
    if "{r1}-{r2}" in prompt:
        no = random.random()
        if no < 0.5:
            p = np.random.choice(["approximately", "roughly", "about", "around", "~"])

            prompt = prompt.replace(
                "{r1}-{r2}", p + " " + str(myround(x["word_count"]))
            )
            return prompt.strip()
    try:
        l1 = np.random.choice([i for i in range(q1, myround(x["word_count"]), 5)])
        l2 = np.random.choice([i for i in range(myround(x["word_count"]), q2, 5)])
        prompt = prompt.replace("{r1}", str(l1)).replace("{r2}", str(l2))
    except:
        prompt = prompt.replace(
            "{r1}", str(max(0, myround(x["word_count"]) - 10))
        ).replace("{r2}", str(myround(x["word_count"]) + 10))
    if "{r2}" in prompt:
        print(prompt)
    if "{r1}" in prompt:
        print(prompt)
    return prompt.strip()


path = "../scored_summarization_datasets/datasets/*"
for dataset, prompts in prompt_dict.items():
    main_dataset = True
    for k in glob.glob(path):
        if dataset in k:
            df = pd.read_parquet(k)
            df["word_count"] = df["summary"].apply(word_count)
            q1 = myround(df["word_count"].quantile(0.05) - 5)
            q_mean = myround(df["word_count"].quantile(0.5))
            q2 = myround(df["word_count"].quantile(0.95) + 5)
            df["prompt"] = df.apply(
                lambda x: get_prompted_comment(
                    x, prompts=prompts, q1=q1, q_mean=q_mean, q2=q2
                ),
                axis=1,
            )
            if main_dataset:
                df_main = df[["prompt", "text","summary",  "contriever_cos"]]
                main_dataset = False
            else:
                df_main = pd.concat(
                    [df_main, df[["prompt", "text", "summary", "contriever_cos"]]]
                )
    df_main.reset_index(inplace=True, drop=True)
    df_main = df_main.drop_duplicates(subset=["text", "summary"])
    df_main.reset_index(inplace=True, drop=True)
    df_main.to_parquet(f"{dataset}_prompted.parquet")
    break

# l = ["approximately", "roughly", "about", "around", "~"]
# V = df_main[df_main['prompt'].apply(lambda x: True if any(i in x for i in l) else False)]
# V.reset_index(inplace=True,drop=True)
from pyarrow.parquet import ParquetFile
import pyarrow as pa

# pf = ParquetFile('/Users/jordanclive/Desktop/datasets/wikipedia-summary-dataset/df_withoutDescription.parquet')
# first_ten_rows = next(pf.iter_batches(batch_size = 10000))
# df = pa.Table.from_batches([first_ten_rows]).to_pandas()
# print("Dataset Length:", len(df))
# print("\n\ntext Word Count")
# print_quantiles(df["full_text"].apply(word_count))
# print("\n\nSummary Word Count")
# print_quantiles(df["summary"].apply(word_count))
# # print('\n\ntext Token Count')
# # print_quantiles(df['text'].apply(token_count))
# # print('\n\nSummary Token Count')
# # print_quantiles(df['summary'].apply(token_count))
#
# print("\n\ntext token calc Count")
# print_quantiles(1.42 * df["full_text"].apply(word_count))
# print("\n\nSummary token calc Count")
# print_quantiles(1.42 * df["summary"].apply(word_count))
# print("\n\ntext Token Count")

# df = pd.read_parquet("/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/scored_summarization_datasets/datasets/billsum_train_scored.snappy.parquet")
# print("Dataset Length:", len(df))
# print("\n\ntext Word Count")
# print_quantiles(df["text"].apply(word_count))
# print("\n\nSummary Word Count")
# print_quantiles(df["summary"].apply(word_count))
# # print('\n\ntext Token Count')
# # print_quantiles(df['text'].apply(token_count))
# # print('\n\nSummary Token Count')
# # print_quantiles(df['summary'].apply(token_count))
#
# print("\n\ntext token calc Count")
# print_quantiles(1.42 * df["text"].apply(word_count))
# print("\n\nSummary token calc Count")
# print_quantiles(1.42 * df["summary"].apply(word_count))
# print("\n\ntext Token Count")
