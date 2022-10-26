## Appendix code for: Biomedical NER for the Enterprise with Distillated BERN2 and the Kazu Framework 
#### - Training and evaluating multi-label NER model (using soft-label setting) 

<br>

This repository presents the train and evaluation codes for the NER module used in the initial release of the KAZU (Korea University and AstraZeneca) framework.
* For the framework, please visit https://github.com/AstraZeneca/KAZU
* For details about the model, please see our paper entitled `Biomedical NER for the Enterprise with Distillated BERN2 and the Kazu Framework`.

Our models are available on Hugging Face framework: 
* [**KAZU-NER-module-distil-v1.0**](https://huggingface.co/dmis-lab/KAZU-NER-module-distil-v1.0): NER module for KAZU framework. Denoted as TinyBERN2 model in the paper. 
* TinyPubMedBERT (will be shortly available)



### Citation info
Joint-first authorship of **Richard Jackson** and **WonJin Yoon**.
<br>Please cite: (Full citation info will be announced soon)
```
@inproceedings{YoonAndJackson2022BiomedicalNER,
  title={Biomedical NER for the Enterprise with Distillated BERN2 and the Kazu Framework},
  author={Wonjin Yoon, Richard Jackson, Elliot Ford, Vladimir Poroshin, Jaewoo Kang},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022}
}
```

## How to train or evaluate a model.

### Requirements

Codes are tested using Python v3.7.13 and following libraries. 
```
torch==1.8.2
transformers==4.9.2
datasets==1.18.3
seqeval>=1.2.2
```

### Example codes
(Example CLI codes are in `example_run_ner.sh`.)

The following steps will provide a simple tutorial on how to produce predictions (and checkpoints if you are trying to train a model) in `${OUTPUT_DIR}`. All the codes are written in Linux bash script and tested on Ubuntu. 

#### Dataset preparation

First, collect CoNLL format datasets. 
<br>In this tutorial, we will download pre-processed BC5CDR benchmark dataset from [cambridgeltl/MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/blob/master/data/BC5CDR-IOB) repository.
```bash
export DATA_DIR=BC5CDR_test 
mkdir -p ${DATA_DIR}
wget -O ${DATA_DIR}/test.tsv https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/BC5CDR-IOB/test.tsv
```

For the inference (eval time), the model is required to use `labels.txt` identical to the one used for training. Please download `labels.txt` from [here](https://huggingface.co/dmis-lab/KAZU-NER-module-distil-v1.0/resolve/main/labels.txt) and place it in `${DATA_DIR}` folder with train or test files.
```
wget -O ${DATA_DIR}/labels.txt https://huggingface.co/dmis-lab/KAZU-NER-module-distil-v1.0/resolve/main/labels.txt
```

**Important:** please make it sure that all the tags in dataset (`test.tsv`) should be in `labels.txt` (case-sensitive). If not, we suggest to modify dataset to match tags in labels.txt. You can replace some non-supported tags (entity types) with `O` tags. 
<br>For example the following bash script can alter the tags in the dataset (works in sed (GNU sed) 4.2.2 - in some sed versions this solution may not work). Alternativly, we suggest you to write a simple python code to change tags.
```bash
#sed -i 's/search_string/replace_string/' filename
sed -i 's/\tB-Disease/\tB-disease/' ${DATA_DIR}/test.tsv
sed -i 's/\tI-Disease/\tI-disease/' ${DATA_DIR}/test.tsv
sed -i 's/\tB-Chemical/\tB-drug/' ${DATA_DIR}/test.tsv
sed -i 's/\tI-Chemical/\tI-drug/' ${DATA_DIR}/test.tsv
```

Final step of dataset preparation is to transform conll format dataset to `*.prob_conll` format. `label2prob.py` is a simple script to change the format.
```bash
export IS_IO="" # set this for using IO taggings.
python label2prob.py --label ${DATA_DIR}/labels.txt  --file_path ${DATA_DIR}/test.tsv --output_path ${DATA_DIR}/test.prob_conll ${IS_IO}
```
Check that the final product has the same number of lines as the original dataset (unless you saw `n` duplicated empty lines from the stdout message of the previous step).
```bash
wc -l ${DATA_DIR}/*
```

Please prepare dev set using the same procedure. 

#### How to eval KAZU-NER model

```bash
export BATCH_SIZE=256
export GRAD_ACCUMLATION=4 
export LEARN_RATE=3e-5
export SAVE_STEPS=100

# Create basic folders
export CACHE_DIR=_tmp/cache
rm -rf ${CACHE_DIR}
mkdir -p ${CACHE_DIR}
export OUTPUT_DIR=_tmp/output/MultiLabelNER-test
mkdir -p ${OUTPUT_DIR}

# To eval (without training)
export BERT_MODEL="dmis-lab/KAZU-NER-module-distil-v1.0"

python3 run_ner.py \
 --model_name_or_path $BERT_MODEL \
 --max_length 128 \
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
```


#### How to train your own model using the code (multi-label NER setting)
```bash
export BATCH_SIZE=256
export GRAD_ACCUMLATION=4 
export LEARN_RATE=3e-5
export SAVE_STEPS=100

# Create basic folders
export CACHE_DIR=_tmp/cache
rm -rf ${CACHE_DIR}
mkdir -p ${CACHE_DIR}
export OUTPUT_DIR=_tmp/output/MultiLabelNER-test
mkdir -p ${OUTPUT_DIR}

# To train
export BERT_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" # or one of our distillated models
#export BERT_MODEL="TinyPubMedBERT" # an example of our distillated model

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

```




### Contact Information
For help or issues using the codes or model (NER module of KAZU) in this repository, please contact WonJin Yoon (`wonjin.info (at) gmail.com`) or submit a GitHub issue.

