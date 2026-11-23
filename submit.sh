#!/bin/sh
### LSF batch job script for DTU HPC (Group 78)
#BSUB -q gpua100
#BSUB -J retrieval_group78
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o outputs/gpu_%J.out
#BSUB -e outputs/gpu_%J.err

set -e

# Create necessary directories
mkdir -p outputs checkpoints/bert_small outputs/attention

echo "=== System and GPU Info ==="
echo "Node: $(hostname)"
nvidia-smi

echo "=== Loading Environment Modules ==="
module load python3/3.10.13
module load cuda/12.1

# DTU HPC python3 module sets PYTHONHOME and PYTHONPATH, which conflict with virtual environments.
unset PYTHONHOME
unset PYTHONPATH

echo "=== Activating Virtual Environment ==="
source .venv/bin/activate

# Remove stale document embedding cache before evaluation
rm -f checkpoints/document_embeddings.pt checkpoints/bert_small/document_embeddings.pt

echo "=========================================================="
echo " 1. Train Baseline DPR (20 epochs, scratch Transformer)   "
echo "=========================================================="
python3 train.py device=cuda

echo "=========================================================="
echo " 2. Train DPR from Pretrained BERT-Small (10 epochs)       "
echo "=========================================================="
python3 train_bert.py device=cuda

echo "=========================================================="
echo " 3. Evaluate Retrievers on FiQA                           "
echo "=========================================================="
echo ">>> (A) TF-IDF <<<"
python3 main.py retriever=tf_idf

echo ">>> (B) BM25 <<<"
python3 main.py retriever=bm25

echo ">>> (C) DPR Baseline <<<"
python3 main.py retriever=dpr device=cuda

echo ">>> (D) DPR Initialized from BERT-Small <<<"
python3 main.py retriever=dpr_bert model=bert_small device=cuda

echo "=========================================================="
echo " 4. Generate Attention Visualization Maps                 "
echo "=========================================================="
python3 visualize_attention.py --dataset fiqa --example 0

echo "=== All tasks completed successfully ==="
