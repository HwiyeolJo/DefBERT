# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch

# Custom datasets
# import load_dataset
from datasets import load_dataset
from datasets import Dataset, DatasetDict
import numpy as np

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, IntervalStrategy
)
from transformers.trainer_utils import is_main_process
from transformers import PreTrainedTokenizer

# from trainer import MyTrainer as Trainer
import logging
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import utils
from tqdm import tqdm
import re

import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import BatchEncoding, PreTrainedTokenizerBase

from datasets.utils.logging import set_verbosity_error

# InputDataClass = NewType("InputDataClass", Any)
# DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
#         idx = torch.nonzero(inputs == self.tokenizer.convert_tokens_to_ids("$$"))
#         for (x1, y1), (x2, y2) in zip(idx[::2],idx[1::2]):
#             masked_indices[x1,y1+1:y2] = True
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices

#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.0)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_hierarchical: bool = field(
        default=False,
        metadata={"help": "Whether to use hierarchical models in decoding."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    ##### Custom
#     PrecedingSentNum: int = field(
#         default=1,
#         metadata={"help": "Num of preceding sentence to be processed."},
#     )
#     SucceedingSentNum: int = field(
#         default=0,
#         metadata={"help": "Num of succeeding sentence to be processed."},
#     )
#     AddNotation: bool = field(
#         default=False,
#         metadata={"help": "Whether latex notations are added."},
#     )
        
    MaxTokenLen: int = field(
        default=512,
        metadata={"help": "Num of maximum token length to be processed."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    ### Orignal Codes
#     else:
#         data_files = {}
#         if data_args.train_file is not None:
#             data_files["train"] = data_args.train_file
#         if data_args.validation_file is not None:
#             data_files["validation"] = data_args.validation_file
#         extension = data_args.train_file.split(".")[-1]
#         if extension == "txt":
#             extension = "text"
#         datasets = load_dataset(extension, data_files=data_files)
    #####
    ### Custom Codes
    else:
#         dataset = load_dataset('json', data_files=data_args.train_file)
        
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["valid"] = data_args.validation_file
            data_files["test"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
#         datasets = load_dataset(extension, data_files=data_files)
#         datasets = data_processor.MathBERTMLM()
#         datasets["train"] = data_processor.MathBERTMLM(data_files["train"], 1)
#         datasets["valid"] = data_processor.MathBERTMLM(data_files["valid"], 1)
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
#             return_special_tokens_mask = True
            return_special_tokens_mask = False
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
#             return_special_tokens_mask = True
            return_special_tokens_mask = False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer,
#             return_special_tokens_mask = True
            return_special_tokens_mask = False,
            bos_token='[CLS]', eos_token='[SEP]', sep_token='[SEP]', cls_token='[CLS]', unk_token='[UNK]',
            pad_token='[PAD]', mask_token='[MASK]',
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
        
#     print(tokenizer.tokenize("[CLS] The den ##omi ##nat ##or comprises the thermal noise power , which is equal to the power spectral density [MASK] multi ##plied by the total bandwidth [MASK] . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"))
#     return
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)
        
    ### Add Vocabulary
    model.resize_token_embeddings(len(tokenizer))
    #####
    ### Get Custom Arguments
#     PrecedingSentNum=data_args.PrecedingSentNum
#     SucceedingSentNum=data_args.SucceedingSentNum
#     AddNotationToVocab=data_args.AddNotation
#     Hierarchical=model_args.use_hierarchical
    ############## Preprocessing the datasets. (Custom)
    PreprocessedData = utils.Preprocessing(tokenizer, data_files["train"])

    train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt = PreprocessedData
        
    if True:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
#             examples["text"] = tokenizer.convert_tokens_to_string(examples["text"].split())
#             examples["text"] = tokenizer.convert_tokens_to_string(examples["text"])
            return tokenizer(
                examples["text"],
#                 padding=padding,
#                 padding="max_length",
                truncation=False,
                max_length=data_args.MaxTokenLen,
#                 max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                add_special_tokens = False,
#                 return_tensors = 'np',
#                 return_special_tokens_mask=True,
                return_special_tokens_mask=False,
            )
        
        DataDict = {
             "train": Dataset.from_dict({ "text": train_tgt }),
             "valid": Dataset.from_dict({ "text": valid_tgt }),
             "test": Dataset.from_dict({ "text": test_tgt  }),
        }
        
        DataDict_test = {
#              "train": Dataset.from_dict({ "text": train_tgt }),
#              "valid": Dataset.from_dict({ "text": valid_tgt }),
             "test": Dataset.from_dict({ "text": test_tgt  }),
        }

        datasets = DatasetDict(DataDict)
        datasets_test = DatasetDict(DataDict_test)

        if training_args.do_train:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["valid"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

#         text_column_name = datasets["train"].column_names[0]
        tokenized_datasets = datasets.map(
#             tokenize_function,
            tokenize_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
#             num_proc=1,
            remove_columns=[text_column_name],
#             load_from_cache_file=not data_args.overwrite_cache,
        )
    
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name],
                             return_special_tokens_mask=False)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    
    # Data collator
    # This one will take care of randomly masking the tokens.
#     data_args.mlm_probability = 0.15
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=data_args.mlm_probability)
    # is masked context hyperparameter
    
    # Training
#     if training_args.do_train:
    if True:
        # Initialize our Trainer
        training_args.overwrite_output_dir=True,
        training_args.do_train=True,
        training_args.do_eval=True,
#         training_args.per_device_train_batch_size=32,
#         training_args.per_device_eval_batch_size=32,
#         training_args.num_train_epochs=10,
#         training_args.logging_steps=100,
#         training_args.logging_first_step=True,
#         training_args.seed=42,
#         training_args.save_strategy=IntervalStrategy.EPOCH,
#         training_args.evaluation_strategy=IntervalStrategy.EPOCH,
#         training_args.load_best_model_at_end=True,
                
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        
        trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        tokenizer.save_vocabulary(training_args.output_dir)
    # Evaluation
    results = {}
    
    if training_args.do_eval:
        # Initialize our Trainer
        training_args.disable_tqdm = False
        
        trainer = Trainer(
            model=model,
            args=training_args,
#             train_dataset=tokenized_datasets["train"],
#             eval_dataset=tokenized_datasets["test"],
#             test_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
#             progress_bar_refresh_rate=0,
        )

        correct, total = 0., 0
        correct_topk = np.zeros(11)
        perplexity_list = []
        logging.disable(sys.maxsize)
        pbar = tqdm(total = len(test_tgt))
        
        training_args.disable_tqdm = True
        set_verbosity_error()
        
        pbar = tqdm(total=len(test_tgt))
        for i in range(len(test_tgt)):
            MaskNum = np.sum(np.array(test_src[i].split()) == "[MASK]")
#             if MaskNum > 10:
#                 pbar.update(1); continue
#             elif MaskNum == 0:
#                 pbar.update(1); continue
            DataDict_test = { "test": Dataset.from_dict({ "text": [' '.join(test_src[i].split()) ] }), }
            datasets_test = DatasetDict(DataDict_test)
            tokenized_datasets_ = datasets_test.map(
                tokenize_function,
                batched=False,
                num_proc=data_args.preprocessing_num_workers,
        #             num_proc=1,
                remove_columns=[text_column_name],
    #                 load_from_cache_file=not data_args.overwrite_cache,
            )

            test_output = trainer.predict(test_dataset=tokenized_datasets_["test"])

            pred_token = test_src[i].split()
            tgt_token = test_tgt[i].split()
            CorrectFlag = False
            if test_output:
                for j in range(len(test_output[1])):
                    p = []
                    for k in range(min(len(test_output[1][j]), len(tgt_token))):
#                             if test_output[1][j][k] != -100: # 
#                                 # To Highlight
#                                 predicted_index = np.argmax(test_output[0][j][k]).item()
#                                 pred_t = tokenizer.convert_ids_to_tokens([predicted_index])[0]
#                                 pred_token[k] = '<<'+ pred_t + '>>'
#                                 pred_token[k] = '<<'+ tokenizer.convert_ids_to_tokens([test_output[1][j][k]])[0] + '>>'

#                             if test_output[1][j][k] != -100: # 
                        if pred_token[k] == "[MASK]":
                            ### Top-1
                            predicted_index = np.argmax(test_output[0][j][k]).item()
                            pred_t = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                            pred_token[k] = '<<' + pred_t + '>>'
                            total += 1

                            if tgt_token[k] == pred_t:
                                correct += 1
                                CorrectFlag = True
                                print("\nCorrect Example", MaskNum)
                                print('IN,', test_src[i].encode('utf-8'))
                                print('Pred,', ' '.join(pred_token).encode('utf-8'), len(pred_token))
                                print('Gold,', test_tgt[i].encode('utf-8'))

                            ### Top-k
                            pred_topk = []; pred_ts = []
                            for n in range(1,10+1): # Starts from Top-1
                                pred_topk = np.argsort(-test_output[0][j][k])[:n].tolist()
#                                 print(pred_topk)
                                pred_ts = tokenizer.convert_ids_to_tokens([t for t in pred_topk])
                                if tgt_token[k] in pred_ts:
                                    correct_topk[n] += 1

                            exp_t = test_output[0][j][k]
                            # https://huggingface.co/transformers/perplexity.html
                            prob = np.exp(exp_t[tokenizer.convert_tokens_to_ids(tgt_token[k])]/len(exp_t))
                            p.append(prob)
#                         print()

#                         if training_args.do_predict:
#     #                         if "$$" in test_tgt[i].split():
#                             if "[MASK]" in test_src[i].split():
#                                 print('IN,', test_src[i].encode('utf-8'))
#                                 print('Pred,', ' '.join(pred_token).encode('utf-8'), len(pred_token))
#                                 print('Gold,', test_tgt[i].encode('utf-8'))
            if CorrectFlag:
                if True:
#                             if "[MASK]" in test_src[i].split():
                    print('IN,', test_src[i].encode('utf-8'))
                    print('Pred,', ' '.join(pred_token).encode('utf-8'), len(pred_token))
                    print('Gold,', test_tgt[i].encode('utf-8'))
                CorrectFlag = False
            if p: perplexity_list.append(np.sum(p))
            pbar.update(1)
        pbar.close()
            
        logging.disable(logging.NOTSET)

        if total == 0:
            print("No MASK in Evaluation Set")
        else:
            print("Top-1 Accuracy", correct, '/', total, correct/total)
            for n in range(1,10+1):
                print("Top-"+str(n), "Accuracy", correct_topk[n]/total)
            print("Perplexity(MASK)", np.mean(perplexity_list))

            eval_output = trainer.evaluate(eval_dataset=tokenized_datasets["test"],)
            perplexity = math.exp(eval_output["eval_loss"])
            print("Perplexity(ALL)", perplexity)

#         perplexity = math.exp(eval_output["eval_loss"])
#         results["perplexity"] = perplexity

#         output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
#         if trainer.is_world_process_zero():
#             with open(output_eval_file, "w") as writer:
#                 logger.info("***** Eval results *****")
#                 for key, value in results.items():
#                     logger.info(f"  {key} = {value}")
#                     writer.write(f"{key} = {value}\n")
        
    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
