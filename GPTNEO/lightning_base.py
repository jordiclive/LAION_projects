import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from transformers import ( AutoConfig, AutoModel,
                          AutoModelForPreTraining,
                          AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoModelWithLMHead,
                          AutoTokenizer, PretrainedConfig, PreTrainedTokenizer, AutoModelForCausalLM)
from transformers.optimization import (
    Adafactor, get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup)
from transformers.utils.versions import require_version
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim import AdamW
import deepspeed

logger = logging.getLogger(__name__)

require_version("pytorch_lightning>=1.0.4")

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
            self,
            hparams: argparse.Namespace,
            config=None,
            tokenizer=None,
            num_labels=None,
            model=None,
            **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.save_hyperparameters(hparams)
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.model_name_or_path = self.hparams.model_name_or_path


        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name
                if self.hparams.tokenizer_name
                else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        if self.hparams.precision == "bf16":
            self.torchtype = torch.bfloat16
        else:
            self.torchtype = torch.float16

    
#         config = AutoConfig.from_pretrained(
#             self.hparams.model_name_or_path)
#         config.gradient_checkpointing = True
        if model is None:
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
#                 config=config,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
#                 revision='6c3dee63e7c7e96311f4b57283b8fd39020a33aa'
            )

        else:
            self.model = model
        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.test_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert (
                self.target_lens["train"] <= self.target_lens["val"]
        ), f"target_lens: {self.target_lens}"
        assert (
                self.target_lens["train"] <= self.target_lens["test"]
        ), f"target_lens: {self.target_lens}"

        # todo complete

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        QA_SPECIAL_TOKENS = {"Question": "<question>", "Answer": "<answer>"}
        self.new_tokens = ['<question>', '<answer>']
        new_tokens_vocab = {}
        new_tokens_vocab["additional_special_tokens"] = []
        for idx, t in enumerate(self.new_tokens):
            new_tokens_vocab["additional_special_tokens"].append(t)
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        rank_zero_info("We have added %s tokens", num_added_toks)

        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]

        scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.opt = AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )

        scheduler = self.get_lr_scheduler()

        return [self.opt], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    # @rank_zero_only
    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from huggingface.co",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.01,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
        )
        parser.add_argument(
            "--num_train_epochs", dest="max_epochs", default=10, type=int
        )
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--adafactor", default=True, type=bool)
        parser.add_argument("--save_path", default="", type=str)
        parser.add_argument("--hf_checkpoint", default=False, type=bool)
        parser.add_argument("--load_checkpoint", default=False, type=bool)
        parser.add_argument("--data_path", default="../data", type=str)
        parser.add_argument("--deepspeed_stage_3", default=False, type=bool)
        parser.add_argument("--eval_beams", default=2, type=bool)


def add_generic_args(parser, root_dir) -> None:
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument(
        "--max_grad_norm",
        dest="gradient_clip_val",
        default=1.0,
        type=float,
        help="Max gradient norm",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument("--logger_name", type=str, default=None)
    parser.add_argument("--wb_name", type=str, default="")
    parser.add_argument("--wb_project", type=str, default="")
    parser.add_argument("--wb_entity", type=str, default="jordanclive")
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--full_test", type=bool, default=False)
    parser.add_argument("--save_path_for_test", type=str, default="")
    parser.add_argument("--val_check_interval", type=float, default=0.15)
    parser.add_argument("--num_sanity_val_steps", type=int, default=-1)
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--limit_val_batches", type=float, default=None)


def generic_train(
        model: BaseTransformer,
        args: argparse.Namespace,
        early_stopping_callback=None,
        logger=True,  # can pass WandbLogger() here
        extra_callbacks=[],
        checkpoint_callback=None,
        logging_callback=None,
        **extra_train_kwargs,
):
    pl.seed_everything(args.seed)

    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=odir,
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
    )

    train_params = {}

    train_params["accelerator"] = 'gpu'
    train_params["accumulate_grad_batches"] = model.hparams.accumulate_grad_batches
    train_params["precision"] = args.precision
    train_params["strategy"] = "deepspeed_stage_2"
    train_params["devices"] = 8

    deepspeed_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "cpu_offload": True
    },
        "zero_allow_untested_optimizer": True,

}
    train_params["strategy"] = DeepSpeedStrategy(config=deepspeed_config)

 




    if model.hparams.deepspeed_stage_3:
        train_params["strategy"] = DeepSpeedPlugin(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            load_full_weights=True,
        )
        train_params["strategy"] = "ddp_sharded"

    train_params["profiler"] = extra_train_kwargs.get("profiler", None)
    train_params["gradient_clip_val"] = args.gradient_clip_val
    train_params["val_check_interval"] = args.val_check_interval
    train_params["num_sanity_val_steps"] = args.num_sanity_val_steps
    if model.hparams.local:
        train_params["precision"] = 32
        train_params["num_sanity_val_steps"] = 10
        train_params["gpus"] = 0
        train_params["strategy"] = None

    if args.limit_val_batches is not None:
        train_params["limit_val_batches"] = args.limit_val_batches

    if args.resume_from_checkpoint is not None:
        train_params["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.debug_mode:
        train_params["limit_train_batches"] = 100
        train_params["limit_val_batches"] = 30
    if args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        if args.id is not None:
            id_ = args.id
        else:
            id_ = wandb.util.generate_id()
        logger = WandbLogger(
            id=id_, name=args.wb_name, project=args.wb_project, entity=args.wb_entity
        )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        **train_params,
    )
    trainer.fit(model)

    return trainer
