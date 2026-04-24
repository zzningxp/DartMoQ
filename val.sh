#!/bin/sh

export HF_DATASETS_OFFLINE=1
# export HF_ENDPOINT=https://hf-mirror.com

python eval_dartmoq.py ~/models/deepseek-moe-16b-base/ --eval-zero
