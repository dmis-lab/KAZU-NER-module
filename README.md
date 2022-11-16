## Multi-label token classification (using soft-label settings) - Modeling for NER task
### Appendix code for: Biomedical NER for the Enterprise with Distillated BERN2 and the Kazu Framework (EMNLP 2022)

<br>

This repository presents the train and evaluation codes for the NER module used in the initial release of the KAZU (Korea University and AstraZeneca) framework.
* For the framework, please visit https://github.com/AstraZeneca/KAZU
* For details about the model, please see our paper entitled `Biomedical NER for the Enterprise with Distillated BERN2 and the Kazu Framework (EMNLP 2022 Industry track)`.

Our models are available on Hugging Face framework: 
* [**KAZU-NER-module-distil-v1.0**](https://huggingface.co/dmis-lab/KAZU-NER-module-distil-v1.0): NER module for KAZU framework. Denoted as TinyBERN2 model in the paper. 
* [**Tiny-sized PubMedBERT-v1.0**](https://huggingface.co/dmis-lab/TinyPubMedBERT-v1.0): Distillated [PubMedBERT (Gu et al., 2021)](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract).
* [**Tiny-sized SapBERT-v1.0**](https://huggingface.co/dmis-lab/TinySapBERT-from-TinyPubMedBERT-v1.0): Tiny-sized (4-layer) biomedical entity representations (language model). Training of this weights are initiated from TinyPubMedBERT-v1.0 and trained using [SapBERT (Liu et al., 2021)](https://github.com/cambridgeltl/sapbert) scheme.

We applied multi-label token classification settings for NER task to mitigate the problem from _nested entity_ recognition (Please check `Section 3.1 Model Architecture` of the paper).

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
We would like to recommend you to additionally cite [PubMedBERT (Gu et al., 2021)](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) or [SapBERT (Liu et al., 2021)](https://github.com/cambridgeltl/sapbert) if you use our release of a distillated version of such models.
<hr>

## How to train or evaluate a model.

### Table of Contents
- [Requirements](#requirements)
- [Example codes](#example-codes)
- [Input data format](#input-data-format)
- [Dataset preparation](#dataset-preparation)
- [How to eval KAZU\-NER model](#how-to-eval-kazu-ner-model)
- [How to train your own model using the code](#how-to-train-your-own-model-using-the-code-multi-label-ner-setting)
- [Known issues](#known-issues)
- [Contact information](#contact-information)

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

### Input data format

![image](https://user-images.githubusercontent.com/13089179/202068143-315bd182-872b-4a1b-b495-efae957d7c82.png)
The above image shows an example of input data. The example sentences are from test datasets of the NCBI-disease corpus [(Doğan et al., 2014)](https://pubmed.ncbi.nlm.nih.gov/24393765/) and we appended automatic annotations from predictions of one of our models.
* Input data should be in tsv format **without a row for column names. In the image, column names are inserted to facilitate explanation**. 
* Analogous to the CoNLL format, each line includes a token (usually a word delimited by blank spaces or special characters: e.g. line 15-17) and annotation information for the token. 
* The datapoints (i.e. the sentences) are separated by a blank line (line 20 in the example).
* The first column refers to tokens, and from the second to the column just before the last one indicates the probability of the token class, in the order of `labels.txt` that should be provided with the model (e.g. see [here](https://huggingface.co/dmis-lab/KAZU-NER-module-distil-v1.0/blob/main/labels.txt) for the KAZU-NER model). 
* The last column indicates a token class with the highest probability **which is for reference only and not used for the training**.
* We provide scripts for the transformation from CoNLL format. Please see the next section (**Dataset preparation**). For test/inference time, if you do not have true labels, you can put `O` for all tokens, and the model will provide its prediction. Please be informed that in this case, you cannot evaluate the model with your input data. 

As you can see in line number 8-11, this input format can train the model with multiple classes for a token. 
<br>In the example, `ataxia-telangiectasia` is indicated as a disease type entity and at the same time `ataxia-telangiectasia gene` is a gene type entity. (We apply a threshold value of 0.5 throughout the repo)

### Dataset preparation

First, collect CoNLL format datasets. 
<br>In this tutorial, we will download pre-processed BC5CDR benchmark dataset from [cambridgeltl/MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/blob/master/data/BC5CDR-IOB) repository.
```bash
export DATA_DIR=${HOME}/KAZU-NER-exp/BC5CDR_test # Please use absolute path to avoid some unexpected errors 
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

### How to eval KAZU-NER model

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

You will see evaluation results from stdout:
```
### Eval results:
# cell_type   : precision: 0.0000, recall: 0.0000, f1: 0.0000, number: 0.0000, accuracy: 0.9967
# drug        : precision: 0.9452, recall: 0.8655, f1: 0.9036, number: 5383.0000, accuracy: 0.9892
# disease     : precision: 0.7691, recall: 0.6625, f1: 0.7118, number: 4424.0000, accuracy: 0.9744
# gene        : precision: 0.0000, recall: 0.0000, f1: 0.0000, number: 0.0000, accuracy: 0.9810
# cell_line   : precision: 0.0000, recall: 0.0000, f1: 0.0000, number: 0.0000, accuracy: 0.9984
# species     : precision: 0.0000, recall: 0.0000, f1: 0.0000, number: 0.0000, accuracy: 0.9945
```
The predictions (labels for tokens) are written in `${OUTPUT_DIR}/predictions.txt`.

### How to train your own model using the code (multi-label NER setting)
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

### Known issues

* **FileNotFoundError: Unable to find '\<FILE\>' at \<LOCATION\>/KAZU-NER-module/prob_conll** :
  <br>This error occurs when one of `--train_file`, `--validation_file`, or `--test_file` is missing from the location you pass through the command line argument.
There is a minor issue about error log especially about the dataset folder path and we are working on this. The error message is sometimes not directly related to the error. 
We suggest you to double check whether the required datasets (i.e. the location you pass to the script) are actully exist in the location (from our example codes,  ${DATA_DIR} folder).


### Contact information
For help or issues using the codes or model (NER module of KAZU) in this repository, please contact WonJin Yoon (`wonjin.info (at) gmail.com`) or submit a GitHub issue.

