# This code generates "CoNLL with probs" format dataset from standard (single-label) CoNLL format dataset.
# Can be used to transform some benchmark dataset (dev or test) to multi-label datasets.

import argparse
from copy import deepcopy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--file_path', type=str, default="resources/dev.txt",  help='')
parser.add_argument('--output_path', type=str, default="resources/dev.conll_prob", help='folder')
parser.add_argument('--label', type=str, default="resources/labels.txt", help='folder')
parser.add_argument('--is_io', action='store_true', help='')
args = parser.parse_args()

label2idx = dict()
with open(args.label, "r") as labFP:
    for line_idx, line in enumerate(labFP.readlines()):
        label2idx[line.splitlines()[0]] = line_idx
if args.is_io:
    ext_label2idx = dict()
    for lab, idx in label2idx.items():
        ext_label2idx[lab] = idx
        new_lab = lab.replace("I-", "B-")
        ext_label2idx[new_lab] = idx
else:
    ext_label2idx = deepcopy(label2idx)

with open(args.file_path, 'r') as inpFP, open(args.output_path, 'w') as outFP:
    DOUBLE_LINE_FLAG = False
    FLAG_count = 0

    for line in inpFP.readlines():
        plain_line = line.splitlines()[0]
        if plain_line.strip() == "":
            if DOUBLE_LINE_FLAG == True:
                FLAG_count += 1 # skip writting duplicated empty lines
            else:
                outFP.write("\n")
            DOUBLE_LINE_FLAG = True
        else:
            DOUBLE_LINE_FLAG = False
            input_line_parsed = plain_line.replace(" ", "\t").split("\t")
            assert len(input_line_parsed) == 2, "len(input_line_parsed) should be 2. Currently: plain_line:%s, input_line_parsed: %s"%(plain_line, input_line_parsed) 
            token = input_line_parsed[0]
            label = input_line_parsed[-1].replace("B-", "I-") if args.is_io else input_line_parsed[-1]
            label_probs = ["0.0"]*len(label2idx) # this is a list of values (probs)
            label_probs[ext_label2idx[label]] = "1.0"
            outSting = "\t".join([token] + label_probs + [label])
            outFP.write(outSting+"\n")

print("Done with %s duplicated empty lines."%(FLAG_count))