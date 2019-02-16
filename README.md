# pytorch-bert-fine-tuning
Fine tuning runner for BERT with pytorch.

Used the files from [huggingface/pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT)

`modeling.py`: Downloaded the pretrained bert to save time, and changed the directory due to proxy problems. Also, `modeling.py` in this repo contains the modeling structure for **multi-task learning with GLUE dataset**. For single task learning, use `modeling.py` directly from [huggingface/pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT).

## Single Task Learning
1. `run_classifier.py` - Fine tuning for GLUE dataset

The original repo only worked only for CoLA, MNLI, MRPC datasets. I added other processors for other remaining tasks as well, so it will work for other tasks, if given the correct arguments.

There was a problem for STS-B dataset, since the labels were continuous, not discrete. I had to create a variable `bin` to adjust the number of final output labels.

Example for QNLI task

```
python3 examples/run_classifier.py \
  --task_name QNLI \
  --do_train  \
  --do_eval  \
  --do_lower_case  \
  --data_dir ./glue_data/QNLI/  \    # Data directory 
  --bert_model bert-base-uncased \
  --max_seq_length 128  \
  --train_batch_size 16  \
  --learning_rate 2e-5  \
  --num_train_epochs 3.0  \
  --output_dir ./tmp/qnli_output/    # Output directory
```

2. `run_squad.py`, `run_squad2.py` - Fine tuning for SQuAD v1.1, v2.0 dataset

3. `run_ner.py` - Fine tuning for CoNLL 2003 dataset (Named Entity Recognition)

`_read_data` function in `DataProcessor` will parse the dataset file. After reading the data, tokenize it with the given tokenizer.
But since the length after tokenization (number of total tokens) does not equal the number of words in the original sentence, I needed to label the new tokens. 

To implement this, I compared each word from the original text with each word from the tokenized text. If it equals, pass. If it doesn't, I took the next word from the tokenized text and (slice `##` if necessary) append to the current word from the tokenized text, and then compare until a match is found.

The tokens that start with `##` were given the label `X`. i.e. Do not make predictions for those tokens.

Example
```
python3 examples/run_ner.py \
  --do_train  \
  --do_eval \
  --do_lower_case  \
  --data_dir ./CoNLL/  \            # Data directory
  --bert_model bert-base-uncased \
  --max_seq_length 32  \
  --train_batch_size 16  \
  --learning_rate 2e-5  \
  --num_train_epochs 3.0  \
  --output_dir ./tmp/ner_output/    # Output Directory
```

I cannot remember the exact results, but the F1 score using BERT-base was greater than 92 for the dev set.

For some reason, `exact_match` was very low.

4. `run_swag.py` - Fine tuning for SWAG dataset

## Multi Task Learning 
* `run_mult.py` - Multi Task Learning for GLUE datasets

Due to lack of time, I only tried multi-task learning for GLUE datasets.
I implemented only for multi-tasking 2 tasks, but most of the variables were modified to be python lists, hoping them to be flexible for tasks more than 2.

1. Model Architecture

To implement this, I used **hard parameter sharing**. Set the pretrained bert as the bottom layer and create an additional layer for each task. The parameters in the bert layer will be shared, but the parameters on the additionals layers for each task will not be shared.

In `modeling.py`, the class `BertForSequenceClassification` was modified. It will only have `BertModel` configs and dropout probability. Also, its `forward` method will return `pooled_output`, which will be passed on to the next layer, depending on the task.

I created a new class `GlueModel` for the additional layer that will be given for each tasks. Its `forward` method will take the `pooled_output` from the previous layer and calculate the loss (or logits).

2. Training

For every training step, the program will choose among given tasks randomly and train (applying back propagation for each step) the model. (Currently) I had planned to adjust the probabilities of choosing each task proportional to the batch size of each task, (Which would result in a multinomial distribution) but this is not implemented yet.

Another unsolved problem is that an epoch doesn't end until all task has used up all its training data. This can be a problem if the data size for each task is different. For instance, if the training data for task A was compeletely selected, then for the rest of the epoch, only the data for task B will be selected. This would cause the shared bert model to forget the parameters that were trained for task A.

Solutions to this problem would be to modifiy when an epoch would end, or keep providing the training data for the task that used up all of its data.

These questions are still left unanswered:
* In which order should we tackle the various tasks?
* Should we switch tasks periodically?
* Should all the tasks be trained for the same number of epochs?

**Unsolved: The model may not save after training.**

3. Changes to program arguments
This file was modified from `run_classifier.py` so the arguments are similar.

Compared to `run_classifier.py`:
* Need to pass 2 task names - `--task1_name`, `--task2_name`
* Need to pass 2 data directories - `--data_dir1`, `--data_dir2`
* Able to set different batch sizes for each task - `--train_batch_size1`, `--train_batch_size2`
* `--do_eval` is not a flag and takes an decimal integer as argument. This decimal integer should be decided by using one-hot encoding representation for each task, from the lowest significant bit. For exmaple, 2, which is 10 in binary, will mean *do not evaluate task 1, evaluate task 2*.

Example:
```
python3 examples/run_mult.py \
  --task1_name QNLI \                 # Task names
  --task2_name MNLI \                 
  --data1_dir ./glue_data/QNLI/  \    # Data directories
  --data2_dir ./glue_data/MNLI/  \    
  --do_train  \
  --do_eval 3 \                       # 3 = 11(2) - Evaluate both tasks
  --do_lower_case \
  --bert_model bert-base-uncased \
  --max_seq_length 128  \
  --train_batch_size1 16  \           # Training batch sizes
  --train_batch_size2 32  \
  --learning_rate 2e-5  \
  --num_train_epochs 3.0  \
  --output_dir ./tmp/qnlimnli_output/     # Output directory
```

Unfortunately due to the training problem mentioned above (and lack of time), the evaluation results were not improved compared to single tasked learning.
