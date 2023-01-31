import pytorch_lightning as pl
import torch


class CustomDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        tokenizer,
        input_length,
        output_length,
        print_text=False,
        val = False,
        num_samples=None,
    ):
        self.source = dataset.text
        self.target = dataset.summary
        self.dataset = dataset
        # if num_samples:
        #     self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
        self.val = val
        if val:
            self.ids = list(range(len(dataset)))

    def __len__(self):
        return len(self.source)

    def clean_text(self, text):
        text = text.strip()
        # text = text.replace('Example of text:', '')
        # text = text.replace('Example of Summary:', '')
        # text = text.replace("\n", "")
        # text = text.replace("``", "")
        # text = text.replace('"', "")

        return text

    def convert_to_features(self, source, target):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(source))

        input_ = self.clean_text(source)
        input_ = 'Summarize the following: '+input_.strip()

        target_ = self.clean_text(target)

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(
            self.source[index], self.target[index]
        )

        input_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if self.val:
            ids = self.ids[index]
            return {
            "input_ids": input_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
            "target_mask": target_mask,
            "index": ids
        }
        else:

            return {
                "input_ids": input_ids,
                "attention_mask": src_mask,
                "labels": target_ids,
                "target_mask": target_mask,
            }


