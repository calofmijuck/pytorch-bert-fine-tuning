# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, GlueModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except Exception as e:
                continue
        return examples

# variable bin as the number of bins
bin = 10

class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # Data labeled in real numbers... what to do here?
    def get_labels(self):
        """See base class."""

        return [str(i) for i in np.linspace(0, 5, bin + 1)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                guid = "%s-%s" % (set_type, i)
                text_a = line[7]
                text_b = line[8]
                label = str(round(float(line[9]) * (bin / 5), 0) / (bin / 5)) # does not work properly...
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except Exception as e:
                continue
        return examples

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            except Exception as e:
                continue
        return examples

class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except Exception as e:
                continue
        return examples

class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                text_b = line[4]
                label = line[5]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except Exception as e:
                continue
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # Data Directory
    parser.add_argument("--data_dir1",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir2",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # Bert Model
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    # Name of Task 1
    parser.add_argument("--task1_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the first task to train.")
    # Name of Task 2
    parser.add_argument("--task2_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the second task to train")
    # Output Directory
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    # Max sequence length
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # Train it?
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    # Run evaluation?
    parser.add_argument("--do_eval",
                        default=0,
                        type=int,
                        help="Whether to run eval on the dev set. 0: Don't eval, 1: Eval task 1, 2: Eval task 2, 3: Eval both")
    # Uncased?
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Set batch size for the first task
    parser.add_argument("--train_batch_size1",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    # Set batch size for the second task
    parser.add_argument("--train_batch_size2",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    # Batch size for evaluation
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    # Learning Rate for Adam
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    # Training epochs
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    # ??
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    # Select Processor
    processors = {
        "rte": RteProcessor,
        "stsb": StsbProcessor,
        "sst2": Sst2Processor,
        "qnli": QnliProcessor,
        "qqp": QqpProcessor,
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
    }

    # number of labels for each task
    num_labels_task = {
        "rte": 2,
        "stsb": bin + 1,
        "sst2": 2,
        "qnli": 2,
        "qqp": 2,
        "cola": 2,
        "mnli": 3,
        "mrpc": 2,
    }

    # cuda or cpu
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # Check for valid args
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # Set train batch size
    args.train_batch_size1 = int(args.train_batch_size1 / args.gradient_accumulation_steps)
    args.train_batch_size2 = int(args.train_batch_size2 / args.gradient_accumulation_steps)

    train_batch_size = [args.train_batch_size1, args.train_batch_size2]


    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and args.do_eval == 0:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # Set task name
    task_name = [args.task1_name.lower(), args.task2_name.lower()]

    # Check if task is in processors. Will need to add to the dictionary if I plan to add another task
    for i in range(2):
        if task_name[i] not in processors:
            raise ValueError("Task %d not found: %s" % (i + 1, task_name[i]))

    # Run the processor. Will need to check what each processor does
    # Create each processor
    processor = [processors[task_name[0]](), processors[task_name[1]]()]

    # Task label
    num_labels = [num_labels_task[task_name[0]], num_labels_task[task_name[1]]]

    # List of labels
    label_list = [processor[0].get_labels(), processor[1].get_labels()]

    # Call Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = []
    num_train_steps = []
    data_dir = [args.data_dir1, args.data_dir2]

    # Train ?
    if args.do_train:
        for i in range(2):
            train_examples.append(processor[i].get_train_examples(data_dir[i]))
            num_train_steps.append(int(len(train_examples[i]) / train_batch_size[i] / args.gradient_accumulation_steps * args.num_train_epochs))

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
              # Do I need num_labels as a parameter?
    ## Need to modify the model in modeling.py... How to?
    mconfig = model.config
    multilayer = [GlueModel(mconfig, num_labels[i]) for i in range(2)]
    for e in multilayer:
        e.cuda()

    ## Additional Optimizers

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer TODO
    param_optimizer = [list(model.named_parameters()) + list(multilayer[i].named_parameters()) for i in range(2)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [[
        {'params': [p for n, p in param_optimizer[i] if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer[i] if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] for i in range(2)]
    t_total = sum(num_train_steps)
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        # Create Optimizer
        ## Todo
        optimizer = [BertAdam(optimizer_grouped_parameters[i],
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total) for i in range(2)]

    global_step = 0
    nb_tr_steps = [0, 0]
    tr_loss = [0, 0]
    train_features = []
    train_data = []
    train_sampler = []
    train_dataloader = []
    if args.do_train:
        logger.info("***** Running training *****")
        for i in range(2):
            train_features.append(convert_examples_to_features(
                train_examples[i], label_list[i], args.max_seq_length, tokenizer))
            logger.info("  Task %d", i + 1)
            logger.info("  Num examples = %d", len(train_examples[i]))
            logger.info("  Batch size = %d", train_batch_size[i])
            logger.info("  Num steps = %d", num_train_steps[i])
            all_input_ids = torch.tensor([f.input_ids for f in train_features[i]], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features[i]], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features[i]], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features[i]], dtype=torch.long)
            train_data.append(TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids))

            if args.local_rank == -1:
                train_sampler.append(RandomSampler(train_data[i]))
            else:
                train_sampler.append(DistributedSampler(train_data[i]))
            train_dataloader.append(list(DataLoader(train_data[i], sampler=train_sampler[i], batch_size=train_batch_size[i])))

        model.train() ## apply for each
        for layer in multilayer:
            layer.train()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = [0, 0]
            nb_tr_examples = [0, 0]
            nb_tr_steps = [0, 0]
            step = [0, 0]
            length = [len(e) for e in train_dataloader]
            for g_step, _ in enumerate(tqdm(range(0, sum(length)), desc="Iteration")):
                if not any([step[i] - length[i] for i in range(2)]):
                    break ## loop finished, added just in case
                elif step[0] == length[0]:
                    select = 1
                elif step[1] == length[1]:
                    select = 0
                else:
                    select = random.randint(0, 1)
                ## Batch size ratio is not taken into consideration

                batch = train_dataloader[select][step[select]]
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                pooled_output = model(input_ids, segment_ids, input_mask, label_ids)
                loss = multilayer[select].foward(pooled_output, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss[select] += loss.item()
                nb_tr_examples[select] += input_ids.size(0)
                nb_tr_steps[select] += 1
                if (g_step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer[select].param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer[select].step()
                    optimizer[select].zero_grad()
                    global_step += 1
                step[select] += 1 ## Index check


    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    layers_to_save = [multilayer[i].module if hasattr(multilayer[i], 'module') else model for i in range(2)]
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_layer_file = [os.path.join(args.output_dir, "pytorch_model_layer%d.bin" % i) for i in range(2)]
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
        for i in range(2):
            torch.save(layers_to_save[i].state_dict(), output_layer_file[i])

    # # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict) # ...? This also needs to be modified
    #
    # # model.load_state_dict(torch.load(path))
    # #     with open(path, 'wb') as f:
    # #         torch.save(model.state_dict(), f)
    # multilayer = [GlueModel.load_state_dict(torch.load(output_layer_file[i])) for i in range(2)]
    # model.to(device)


    eval_flag = args.do_eval

    if eval_flag != 0 and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        task_cnt = 0
        while eval_flag > 0:
            if not eval_flag & 1:
                eval_flag >>= 1
                task_cnt += 1
                continue
            eval_examples = processor[task_cnt].get_dev_examples(data_dir[task_cnt])
            eval_features = convert_examples_to_features(
                eval_examples, label_list[task_cnt], args.max_seq_length, tokenizer)
            logger.info("***** Running evaluation for Task %d*****", task_cnt + 1)
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            multilayer[task_cnt].eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_pooled_output = model(input_ids, segment_ids, input_mask, label_ids)
                    tmp_eval_loss = multilayer[task_cnt].foward(tmp_pooled_output, label_ids)
                    pooled_output = model(input_ids, segment_ids, input_mask)
                    logits = multilayer[task_cnt].foward(pooled_output)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss[task_cnt]/nb_tr_steps[task_cnt] if args.do_train else None
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': loss}

            output_eval_file = os.path.join(args.output_dir, "eval_results%d.txt" % (task_cnt + 1))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results for Task %d*****", task_cnt)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            eval_flag >>= 1
            task_cnt += 1

# Calls Main function
if __name__ == "__main__":
    main()
