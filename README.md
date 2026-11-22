# 02456 Retrieval Project - Group 78

Project 4 in 02456 Deep Learning (Fall 2026). We compare TF-IDF, BM25 and DPR on FiQA and SQuAD. For DPR we train a baseline with our own Transformer encoder and a version initialised from BERT-Small.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

Evaluate a retriever (prints recall, precision and MRR for train, val and test):

```bash
python main.py retriever=tf_idf
python main.py retriever=bm25
python main.py retriever=dpr device=cuda
python main.py retriever=dpr_bert model=bert_small device=cuda
```

Train the baseline DPR or DPR from BERT-Small (checkpoints are saved in `checkpoints/`):

```bash
python train.py device=cuda
python train_bert.py device=cuda
```

Add `dataset=squad` to any command to use SQuAD instead of FiQA. The document embeddings are saved at `retriever.index_path` and reused, so delete that file after retraining.

Plot attention maps of the trained encoders (saved in `outputs/attention/`):

```bash
python visualize_attention.py --dataset squad --example 0
```

## Experiments

All experiments were run on both FiQA and SQuAD:

- TF-IDF and BM25 with `main.py`.
- Baseline DPR: `train.py` with the defaults in `config/train_dpr.yaml`.
- DPR from BERT-Small: `train_bert.py` with the defaults in `config/train_bert.yaml`.
- Pooling: `train_bert.py` with `model.pooling_type=cls`, `mean` and `max`.

The baseline is trained for 20 epochs and DPR from BERT-Small for 10 epochs. The checkpoint with the best validation in-batch accuracy is kept.

## Structure

```
config/           Experiment, dataset, retriever, model and training settings.
main.py           Evaluates a retriever.
train.py          Trains the baseline DPR.
train_bert.py     Trains DPR initialised from BERT-Small.
visualize_attention.py
src/
├── data_utils/   Dataset loading, corpus, documents, queries and DPR dataloader.
├── retrievers/   TF-IDF, BM25, DPR and the encoders.
├── training/     DPR loss and trainer.
└── evaluation/   Recall, precision and MRR.
```
