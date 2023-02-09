import os
import pandas as pd
from urllib.request import urlopen

from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
QA_SPECIAL_TOKENS = {"Question": "<question>", "Answer": "<answer>"}


@dataclass
class DialogueDataCollator:
    """
    Expects a list of texts corresponding to a sequence of [question, answer, question, answer, ...] pairs.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        flatten_messages = []
        label_masks = []

        for feature_one in features:
            assert len(feature_one) % 2 == 0, "Number of messages must be even"
            messages = [
                # (QA_SPECIAL_TOKENS["Question"] if i % 2 == 0 else "")
                x
                + (QA_SPECIAL_TOKENS["Answer"] if i % 2 == 0 else "")
                for i, x in enumerate(feature_one)
            ]

            # Add a way for the model to terminate generation
            # When we predict the start of a new expected question, we want to be able to stop generation
            messages.append(self.tokenizer.eos_token)

            flatten_message = self.tokenizer(
                "".join(messages),
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )

            message_change_indices = np.cumsum([len(x) for x in messages[:-1]])
            # for each token an integer indicating the index of the message it belongs to. Just to create the label mask.
            # Label mask is true when predicting a token that is part of the answer, false otherwise.
            # TEXT:             Question: Hello, how are you? Answer: I am fine. Question: What is your name? Answer: My name is John. Question:
            # MESSAGE_INDICES:  0         0      0   0   0    0       1 1  1     2         2    2  2    2     2       3  3    3  3     -2
            # LABEL_MASK:       0         0      0   0   0    1       1 1  1     0         0    0  0    0     1       1  1    1  1     0

            # If no result in next, we are predicting the last termination token(s)
            message_indices = list(
                map(
                    lambda x: next((i for i, val in enumerate(message_change_indices) if val >= x), -2),
                    list(map(lambda x: x[1], flatten_message["offset_mapping"])),
                )
            )
            label_mask = np.roll(list(map(lambda x: x % 2 == 1, message_indices)), -1, -1)
            try:
                label_mask[[i for i in range(len(message_indices)) if message_indices[i] == -2][0] - 1] = True
            except IndexError:
                # due to truncation, we might not have the last termination token
                label_mask[-1] = False

            label_masks.append(label_mask)

            flatten_messages.append({k: v for k, v in flatten_message.items() if k != "offset_mapping"})

        batch = self.tokenizer.pad(
            flatten_messages,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        dim = batch["input_ids"].shape[-1]

        batch["label_masks"] = torch.stack(
            [F.pad(torch.tensor(x), (0, dim - len(x)), value=False) for x in label_masks]
        )
        batch["targets"] = torch.roll(batch["input_ids"], -1, -1)

        return batch


class PromptGeneratedDataset(Dataset):
    """Generates from flan 11B
    User: What are the best methods for preventing a slave trade?

    Rosey: The best methods ....
    <|endoftext|>

    we are ignoring results with multiple lines for now
    """

    url = "https://github.com/Rallio67/language-model-agents/raw/main/chat_dialogue_v2_c.txt"

    def __init__(self,df) -> None:
        super().__init__()

        # df = pd.read_json('train.json',orient='split')
        # df.reset_index(inplace=True,drop= True)
        # val = df.sample(frac=0.1,random_state=42)
        # train = df[~df.index.isin(val.index)]
        # train.reset_index(inplace=True,drop= True)
        # val.reset_index(inplace=True,drop=True)
        # train.to_json('train.json',orient='split')
        # val.to_json('val.json',orient='split')
        self.pairs = list(zip(df['source'], df['target']))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        question, answer = self.pairs[index]
        return question, answer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer


    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # tokenizer.pad_token= "<|padding|>"
    # tokenizer.pad_token_id = 1
    tokenizer.add_special_tokens({"pad_token": "<|padding|>", "sep_token": "<|endoftext|>"})
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # train = pd.read_json('test/train.json',orient='split')
    # dataset = PromptGeneratedDataset(train)
    # collate_fn = DialogueDataCollator(tokenizer, padding=True, max_length=128)
    # train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5)
    # # for batch in dataloader:
    #     print(batch["input_ids"].shape)

    val = pd.read_json('test/val.json',orient='split')
    dataset = PromptGeneratedDataset(val)
    collate_fn = DialogueDataCollator(tokenizer, padding=True, max_length=300)
    val_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5)
    # for batch in dataloader:
    #     print(batch["input_ids"].shape)
    X = next(iter(val_dataloader))
    tokenizer.decode(X['input_ids'][0])
    x = 1
