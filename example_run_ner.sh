#!/bin/bash
#
export BATCH_SIZE=256
export GRAD_ACCUMLATION=4 
export LEARN_RATE=3e-5
export SAVE_STEPS=100

export BERT_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" # or one of our distillated models
#export BERT_MODEL="TinyPubMedBERT" # an example of our distillated model
export DATA_DIR=tiny_resources

# Create basic folders
export CACHE_DIR=_tmp/cache
rm -rf ${CACHE_DIR}
mkdir -p ${CACHE_DIR}
export OUTPUT_DIR=_tmp/output/MultiLabelNER-test
mkdir ${OUTPUT_DIR}

python3 run_ner.py \
 --model_name_or_path $BERT_MODEL \
 --max_length 128 \
 --do_train --train_file ${DATA_DIR}/train.prob_conll \
 --learning_rate ${LEARN_RATE} --num_train_epochs 10 \
 --per_device_train_batch_size ${BATCH_SIZE} \
 --gradient_accumulation_steps $GRAD_ACCUMLATION \
 --do_eval --validation_file ${DATA_DIR}/dev.prob_conll \
 --evaluation_strategy steps --eval_steps 100 \
 --per_device_eval_batch_size 256 \
 --do_predict --test_file ${DATA_DIR}/test.prob_conll \
 --cache_dir ${CACHE_DIR} \
 --preprocessing_num_workers 8 \
 --output_dir ${OUTPUT_DIR} \
 --overwrite_output_dir \
 --save_steps $SAVE_STEPS --save_total_limit 50 \
 --return_entity_level_metrics \
 --use_probs

# --no_cuda \ # Use this for experiments using only CPU (i.e. without GPU)
# --fp16 \ # We cannot use fp16 if torch.nn.BCELoss and sigmoid is used.
