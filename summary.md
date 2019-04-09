# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Introduction

1. Feature-based approach
* Task specific 한 구조는 별도로 존재, pre-trained representations 을 feature 로 사용
* word, sentence, paragraph embeddings
* ELMo (Peters et al., 2018. Deep contextualized word representations)
  * Extract context-sensitive features from a language model

2. Fine-tuning approach
* LM 기반 구조로 pre-train 후, downstream task 모델로 사용하여 fine-tuning
* OpenAI GPT (Radford et al., 2018. Improving language understanding with unsupervised learning)
* BERT

3. Contributions
* Bidirectional pre-training 모델의 중요성 보임
* Eliminate the needs of task-specific architectures
* The state-of-the-art for 11 NLP tasks

## Model Architecture

1. BERT: A multi-layer bidirectional Transformer
* OpenAI GPT: left-to-right Transformer
* ELMo: concatenation of independently trained left-to-right/right-to-left LSTM

2. Two models:
* Transformer blocks: L, hidden size: H, self-attention heads: A
* BERT_BASE: L=12, H=768, A=12, Total Params=110M (OpenAI GPT size)
* BERT_LARGE: L=24, H=1024, A=16, Total Params=340M


## Input Representation
* Token + Segment + Position Embeddings
* WordPiece embeddings with a 30,000 token vocabulary (using sub-word unit)
* 최대 512 tokens 에 대해 learned positional embeddings 사용
* [CLS]: The final hidden state (i.e., output of Transformer) => aggregate sequence representation
* [SEP]: separate sentence pairs with a special token
* The first/second sentence – learned sentence A/B embeddings

## Pretraining Tasks
1. Masked LM (MLM)
* Standard conditional language models
  * Left-to-right model (OpenAI GPT), shallow concat model (ELMo) 에 비해 Bidirectional model 이 powerful
  * But, 일반적인 LM models 은 unidirectional 만 가능 (indirectly “see itself” 때문)

* Masked ML
  * Mask 15% of all WordPiece tokens at randomly
  * Only predict the masked words

* Two disadvantages
  * [MASK] token is never seen during fine-tuning
  * Only 15% of tokens are predicted => Takes more steps to converge

2. Next Sentence Prediction (NSP)
* QA, NLI 와 같이 2개의 문장 간의 relationship 에 대한 task 는 LM 에 의해서 직접적으로 잡히지 않음
* Corpus 에서 A, B 2개의 문장을 고르는데,
  * 50% 의 확률로 next sentence (IsNext)
  * 나머지 50% 확률로 random sentence (NotNext)
* Final pre-trained model 에서 이 task 에 대해 97%-98% 달성
* 특히, QA, NLI task 에 대해 매우 효과적임
* Task #1, #2 에 대해 동일한 Input data 사용하여, jointly 학습 (loss = mean MLM likelihood + mean NSP likelihood)

## Procedure
1. Pre-training Procedure
* BooksCorpus (800M words), English Wikipedia (2,500M words) concatenation
  * OpenAI GPT: only use BookCorpus (800M words)
* Training params
  * Batch size = 256, (128,000 tokens/batch – OpenAI GPT 의 4배)
  * Adam with learning rate of 1e-4, beta1 = 0.9, beta2 = 0.9999, L2 weight decay = 0.1
  * Dropout probability = 0.1 on all layers
  * GELU activation following OpenAI GPT
  * Base 모델은 16 TPU, Large 는 64 TPU 총 각각 4일 걸림

2. Fine Tuning Procedure
* Only new parameters are for a classification layer W
* Parameters that work well across all tasks
  * Batch size: 16, 32
  * Learning rate (Adam): 5e-5, 3e-5, 2e-5
  * Number of epochs: 3, 4

## Experiments
1. GLUE Datasets - MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE


2. SQuAD v1.1
* The Stanford Question Answering Dataset (100k)
* Input question 과 paragraph 를 하나의 packed sequence 로 생각
  * The question -> A embedding
  * The paragraph -> B embedding
* A start vector S, end vector E 가 각각 fine-tuning 때 학습됨
* 모든 paragraph 내 token 에 대하여 start vector S 와 dot product 후 softmax (end 도 동일)
* 3 Epochs, learning rate = 5e-5, batch size = 32
* Consider the constraint: end 가 start 보다 항상 뒤에 와야 한다

3. Named Entity Recognition (NER)
* Token tagging task
* CoNLL 2003 NER Dataset
* 200k training words - Person, Organization, Location, Miscellaneous, Other
* +0.2 better than Cross-View Training with multi-task learning (CVT)

4. SWAG
* The Situations With Adversarial Generations dataset
* 113k sentence-pair
* Decide among four choices the most plausible continuation
* Task-specific vector V -> dot product and softmax

## Abalation Studies
1. Effect of pretraining tasks
* Pre-training Tasks 측면에서 변화를 준 2가지 모델과 비교
* No NSP: Pre-train 시 MLM 만 사용, NSP 사용하지 않음
* LTR & No NSP: Left-to-Right (LTR) LM 으로 학습
  * OpenAI GPT 와 다른 점은 dataset, input representation, fine-tuning scheme
* NSP 제거하면, QNLI, MNLI, SQuAD 에서 성능 떨어짐
* LTR -> Bidirectional 효과 확인

2. Effect of model size
* Number of layers, hidden units, and attention heads
* MRPC Task의 경우, 3,600 개의 labeled training data

3. Effect of number of training steps
* Does BERT really need such a large amount of pre-training?
  * 128,000 words/batch * 1,000,000 steps
  * MNLI Task 에 대하여 500k steps 에 비해 1M steps 가 1.0% 높았음
* Does MLM pre-training converge slower than LTR?
  * The MLM model 은 the LTR model 보다 수렴이 느림
  * Accuracy 가 더 높음

4. Feature-based approach with BERT
* Pre-trained model 에서 fixed features 를 가져와 사용
* BERT 의 파라미터 중에서 아무것도 fine-tuning 하지 않음
* CoNLL-2003 NER 실험 결과
  * BERT 는 fine-tuning, feature-based 모두 사용하기 좋음
