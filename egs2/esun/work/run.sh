#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=0

# Required
# llama2: --hugging_face_model_name_or_path "meta-llama/Llama-2-7b-hf"
# llama3.1: --hugging_face_model_name_or_path "meta-llama/Llama-3.1-8B"
# llama3.2: --hugging_face_model_name_or_path "meta-llama/Llama-3.2-1B"
# Qwen2.5: --hugging_face_model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct"
# hugging_face_model_name_or_path="meta-llama/Llama-2-7b-hf"
# hugging_face_model_name_or_path="meta-llama/Llama-3.2-1B"
hugging_face_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"

train_set="train"
valid_set="dev"
test_sets="test dev"

# asr_config=conf/tuning/train_asr_conformer_llama2_vocab.yaml
# asr_config=conf/tuning/train_asr_conformer_llama3_vocab.yaml
# asr_config=conf/tuning/train_asr+llama3_conformer.yaml
asr_config=conf/tuning/train_asr_conformer_qwen2_vocab.yaml

inference_config=conf/tuning/decode_bs1_ctc0.3.yaml

./asr.sh \
    --lang zh \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 1 \
    --token_type hugging_face \
    --hugging_face_model_name_or_path "${hugging_face_model_name_or_path}" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    "$@"
