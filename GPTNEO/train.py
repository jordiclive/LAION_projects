import csv
import gc
import glob
import logging
import os
import re
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

import configargparse
import nltk
import numpy as np
import pandas as pd
import torch
from lightning_base import BaseTransformer, add_generic_args, generic_train
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from rouge_score import rouge_scorer, scoring
from torch.utils.data import DataLoader
from torch import nn
from data import DialogueDataCollator, PromptGeneratedDataset

logger = logging.getLogger(__name__)
ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

import pickle


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    return "\n".join(nltk.sent_tokenize(x))

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super(CrossEntropyLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target, mask=None):
        if mask is not None:
            mask = mask.view(-1).bool()
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
            input = input[mask]
            target = target[mask]
        return super(CrossEntropyLoss, self).forward(input, target)



def calculate_rouge(
        pred_lns: List[str],
        tgt_lns: List[str],
        use_stemmer=True,
        rouge_keys=ROUGE_KEYS,
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.
    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).
    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys
    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {
            stat: round(getattr(mid, stat), 4)
            for stat in ["precision", "recall", "fmeasure"]
        }
    return new_dict


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class ClassificationTransformer(BaseTransformer):
    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        super().__init__(hparams)

        self.val_losses = []
        self.fin_outputs = []
        self.fin_targets = []
        self.test_fin_outputs = []
        self.val_no = 0
        self.dataset_size = len(
            pd.read_json(
                Path(self.hparams.data_path).joinpath("train.json"),orient='split')
        )
        self.save_path = ""
        self.model_name_or_path = self.hparams.model_name_or_path
        self.metric_names = ROUGE_KEYS
        self.decoder_start_token_id = None

        self.val_metric = (
            self.default_val_metric
            if self.hparams.val_metric is None
            else self.hparams.val_metric
        )

        self.eval_min_length = self.hparams.eval_min_length

        if self.hparams.freeze_embeds:
            rank_zero_info('FREEZING embeddings')
            self.freeze_embeds()
        self.loss_fct = CrossEntropyLoss()

    def _compute_loss(self, model, inputs):

        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask", None))
        #self.tokenizer.decode(inputs['input_ids'][labels_mask])
        logits = outputs.logits

        loss = self.loss_fct(logits, targets, mask=labels_mask)

        return loss, logits, targets, labels_mask

    def _step(self, batch, batch_idx) -> Dict:
        loss, logits, labels, labels_mask = self._compute_loss(self.model, batch)
        # labels[~labels_mask.bool()] = self.pad()
        return loss

    def training_step(self, batch, batch_idx) -> Dict:


        loss, logits, labels, labels_mask = self._compute_loss(self.model, batch)
        # labels[~labels_mask.bool()] = self.pad()

        loss = loss.mean()

        self.log(
            "train_loss",
            float(loss),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # torch.cuda.empty_cache()
        return {"loss": loss}

    def freeze_params(self, model):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        for n, p in self.model.named_parameters():
            if "embed" in n:
                p.requires_grad = False


    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def forward(self, inputs):
        return self.model(**inputs, return_dict=False)


    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def get_dataloader(
            self, mode: str, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        "Load datasets. Called after prepare data."
        rank_zero_info(f"batch_size: {batch_size}")

        if mode == "dev":
            data = pd.read_json(
                Path(self.hparams.data_path).joinpath("val.json"), orient='split')

            dataset = PromptGeneratedDataset(data)
            collate_fn = DialogueDataCollator(self.tokenizer, padding=True, max_length=self.hparams.max_seq_length)
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)

            return dataloader

        if mode == "train":
            data = pd.read_json(
                Path(self.hparams.data_path).joinpath("train.json"), orient='split')

            dataset = PromptGeneratedDataset(data)
            collate_fn = DialogueDataCollator(self.tokenizer, padding=True, max_length=self.hparams.max_seq_length)
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)
            return dataloader

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    # @rank_zero_only
    def validation_step(self, batch, batch_idx):
        if self.hparams.skip_val:
            return {'loss': 1}
        if self.hparams.hf_checkpoint:
            save_path = Path(self.output_dir).joinpath("checkpoint-curr-best")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            raise ValueError("just saving")
        return self._generative_step(batch, batch_idx)

    def _generative_step(
            self, batch: dict, batch_idx=None, dataloader_idx=None
    ) -> dict:

        loss = self._step(batch, batch_idx)

        self.log(
            "val_loss",
            float(loss),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


        return {"loss": loss}

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)

        effective_batch_size = (
                self.hparams.train_batch_size
                * self.hparams.accumulate_grad_batches
                * num_devices
        )

        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @rank_zero_only
    def save_hf(self, path):

        rank_zero_info(path)
        save_path = Path(self.hparams.save_path).joinpath(path)
        self.save_path = save_path
        self.model.save_pretrained(save_path, torch_dtype=self.dtype)
        self.tokenizer.save_pretrained(save_path)

#     def on_save_checkpoint(self, checkpoint):
#         path = f"checkpoint-curr-best_{time.strftime('%Y%m%d_%H%M')}"
#         self.save_hf(path)

#     def on_load_checkpoint(self, checkpoint):
#         state_dict = checkpoint['module']
#         state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
#         checkpoint['state_dict'] = state_dict

    @staticmethod
    def flatten(all_g, col):
        l = [x[col] for x in all_g]
        flat_list = [item for sublist in l for item in sublist]
        return flat_list

    def validation_epoch_end(self, outputs: list) -> dict:
        if self.hparams.skip_val:
            return 0
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=112,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--m1_chip",
            default=False,
            type=bool,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )

        parser.add_argument(
            "--num_labels",
            default=256,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )

        parser.add_argument(
            "--gpus",
            default=0,
            type=int,
            help="The number of GPUs allocated for this, it is by default 0 meaning none",
        )

        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Overwrite the cached training and evaluation sets",
        )
        parser.add_argument("--local", default=False, type=bool)
        parser.add_argument("--visible_devices", default="3", type=str)
        parser.add_argument("--offline", default=False, type=bool)
        parser.add_argument("--test_outputs", default="", type=str)

        parser.add_argument(
            "--max_source_length",
            default=512,  # 1024
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=60,  # 56
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--val_max_target_length",
            default=60,  # 142 # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total validation target length specified foor generation",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=150,  # 142
            type=int,
            help="The maximum total test target length specified for generation",
        )

        parser.add_argument(
            "--n_train",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--n_val",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )
        parser.add_argument(
            "--n_test",
            type=int,
            default=-1,
            required=False,
            help="# examples. -1 means use all.",
        )

        parser.add_argument(
            "--label_smoothing", type=float, default=0.0, required=False
        )
        parser.add_argument("--eval_min_length", type=int, default=10, required=False)
        parser.add_argument("--skip_val", type=bool, default=False, required=False)

        parser.add_argument("--val_metric", type=str, default="rouge2", required=False)
        parser.add_argument(
            "--eval_max_gen_length",
            type=int,
            default=60,
            help="never generate more than n tokens",
        )
        parser.add_argument(
            "--length_penalty",
            type=float,
            default=1.0,
            help="length penalty specified for beam search",
        )

        parser.add_argument(
            "--T5_preamble",
            type=bool,
            default=False,
            required=False,
            help="Add the T5 preamble e.g. Summarize this text.",
        )
        parser.add_argument(
            "--no_repeat_ngram_size",
            type=float,
            default=3,
            help="length penalty specified for beam search",
        )
        parser.add_argument(
            "--freeze_embeds",
            type=bool,
            default=False,
            required=False,
            help="Add the T5 preamble e.g. Summarize this text.",
        )
        parser.add_argument(
            "--generate",
            type=bool,
            default=False,
            required=False,
            help="Add the T5 preamble e.g. Summarize this text.",
        )
        parser.add_argument(
            "--save_generations",
            type=bool,
            default=False,
            required=False,
            help="Add the T5 preamble e.g. Summarize this text.",
        )

        return parser


def main():
    parser = configargparse.ArgumentParser(default_config_files=["config.yaml"])
    add_generic_args(parser, os.getcwd())
    parser = ClassificationTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    if not os.path.exists(args.test_outputs):
        os.makedirs(args.test_outputs)

    if args.local:
        args.gpus = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    if args.offline or args.local:
        os.environ['WANDB_API_KEY'] = 'd8216641d549f9bb3d0c5074baa39e15dfd55030'
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ['WANDB_API_KEY'] = 'd8216641d549f9bb3d0c5074baa39e15dfd55030'

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "/fsx/home-jordiclive/gpt_checkpoints",
            f"{time.strftime('%Y%m%d_%H%M')}",
        )
        try:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
        except:
            pass
    model = ClassificationTransformer(args)

    trainer = generic_train(model, args)

    if args.do_predict:
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(args.output_dir, "checkpoint-epoch=*.ckpt"),
                    recursive=True,
                )
            )
        )
        model = model.load_from_checkpoint(checkpoints[-1])
        return trainer.test(model)


if __name__ == "__main__":
    main()
