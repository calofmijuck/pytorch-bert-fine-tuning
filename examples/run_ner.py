"""BERT finetuning runner."""

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import re
import math

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForTokenClassification # ???
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_pretrained_bert.modeling import ???

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class NerExample(object):
    """A single training/test example for the NER dataset."""
    def __init__(self, guid, text, label = None):
        self.guid = guid
        self.text = text
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"word: {self.text}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


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
    def _read_data(cls, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contents = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contents.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contents) == 0 and words[-1] == '.':
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines

class NerProcessor(DataProcessor):
    """Processor for the CoNLL 2003 data set."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_labels(self):
        return ["X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
        # "[CLS]","[SEP]"]
        ]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[1].lower() # I think this should be lowercase
            label = line[0]
            examples.append(NerExample(guid = guid, text = text, label = label))
        return examples

def preprocess(before, after, label_id):
    ptr = 0
    for idx, orig in enumerate(before):
        if after[idx + ptr] == orig:
            continue
        try:
            flag = False # this flag will check for strings that start with ##
            it = ptr
            currstr = after[idx + it]
            while idx + ptr < len(after):
                it += 1
                if after[idx + it].startswith("##"):
                    flag = True
                    currstr += after[idx + it][2:]
                else:
                    currstr += after[idx + it]
                if currstr == orig:
                    for _ in range(it - ptr): # insert it - ptr elements to the list
                        label_id.insert(idx + ptr + 1, label_id[idx + ptr] if not flag else 0)
                    ptr = it # update ptr
                    break
        except IndexError as e:
            print("Index out of bounds.")
            print(before)
            print(after)
            print(label_id)
            raise IndexError

    if len(after) != len(label_id):
        print(len(after), after)
        print(len(label_id), label_id)
        assert len(after) == len(label_id)
    return label_id

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text)
        currlabel = example.label # get current label
        label_id = [label_map[e] for e in example.label.split(" ")]

        # process tokens that start with ## and label them as "X"
        label_id = preprocess(example.text.split(" "), tokens_a, label_id)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]
            label_id = label_id[: (max_seq_length - 2)]
            # currlabel = " ".join(currlabel.split(" ")[:(max_seq_length - 2)])
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        label_id = [0] + label_id + [0]
        # currlabel = "X " + currlabel + " X"
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        # currlabel += " X" * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if len(label_id) != max_seq_length:
            print(label_id, len(label_id))
            print(tokenizer.tokenize(example.text))
            assert len(label_id) == max_seq_length

        # label_id = [label_map[e] for e in example.label.split(" ")]  + padding ## Is this right?
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("id: %s" % " ".join([str(x) for x in label_id]))
            # logger.info("label: %s (id = %s)" % (currlabel, " ".join([str(x) for x in label_id])))
            # won't need currlabel

        features.append(
            InputFeatures(input_ids = input_ids,
                          input_mask = input_mask,
                          segment_ids = segment_ids,
                          label_id = label_id))
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # Data Directory
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # Bert Model
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
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
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    # Uncased?
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # Set batch size
    parser.add_argument("--train_batch_size",
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
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Call Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)

    train_examples = None
    num_train_steps = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Can I use Token Classification here?
    model = BertForTokenClassification.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), num_labels = num_labels)

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

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
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
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc = "Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = BertForTokenClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    ######################################################################

    outputwrite = open('debug', 'w')





    ######################################################################
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
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
        exact_match, num_sentence = 0, 0
        tp, tn = 0, 0
        fp, fn = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            # print(label_ids)
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)
                ### Edited here!!!! - check dimension
                try:
                    for i in range(args.eval_batch_size):
                        outputwrite.write("Start ----\n")
                        sentence_flag = True
                        for j in range(args.max_seq_length):
                            test_prediction = label_list[int(torch.argmax(logits[i][j]))]
                            outputwrite.write(test_prediction + " ")
                            test_answer = label_list[label_ids[i][j]]
                            outputwrite.write(test_answer)
                            outputwrite.write("\n")
                            if test_prediction == "X":
                                continue
                            if test_prediction not in ["O"]: # Positive
                                if test_answer == test_prediction:
                                    tp += 1
                                    outputwrite.write("True Positive")
                                else:
                                    fp += 1
                                    outputwrite.write("False Positive")
                                    sentence_flag = False
                            else:
                                if test_answer == test_prediction:
                                    tn += 1
                                    outputwrite.write("True Negative")
                                else:
                                    fn += 1
                                    sentence_flag = False
                                    outputwrite.write("False Negative")
                            outputwrite.write("\n")
                        if sentence_flag:
                            exact_match += 1
                            outputwrite.write(" - Exact Match - ")
                        num_sentence += 1
                        outputwrite.write("\n")
                except Exception as e:
                    pass

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

        exact_match = exact_match / num_sentence
        fprecision = tp / (tp + fp)
        frecall = tp / (tp + fn)

        f1 = 2 * fprecision * frecall / (fprecision + frecall)
        result = {'exact_match': exact_match, 'f1': f1, 'precision': fprecision, 'recall': frecall, 'true_pos': tp, 'true_neg': tn, 'false_pos': fp, 'false_neg': fn}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

# Calls Main function
if __name__ == "__main__":
    main()
