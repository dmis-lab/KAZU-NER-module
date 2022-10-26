#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom dataset loading script for CoNLL-like QA (PML-NER: Probability-based multi-label NER training) dataset (tsv)"""

import os
import datasets
from typing import List

logger = datasets.logging.get_logger(__name__)


_CITATION = """Yoon et al."""
_DESCRIPTION = """Dataset builder for probability-based multi-label NER training"""

_TRAINING_FILE = "train.prob_conll"
_TEST_FILE = "test.prob_conll"
_DEV_FILE = "dev.prob_conll"
_LABEL_FILE = "labels.txt"

def get_label_names(label_name_path):
    lable_name_list = []
    with open(label_name_path) as fp:
        for line in fp.readlines():
            if line.splitlines()[0] != "":
                lable_name_list.append(line.splitlines()[0])
    return lable_name_list

class ProbMultiLabelNERConfig(datasets.BuilderConfig):
    """BuilderConfig for PML-NER"""

    def __init__(self, **kwargs):
        """BuilderConfig for PML-NER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ProbMultiLabelNERConfig, self).__init__(**kwargs)


class ProbMultiLabelNERBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        ProbMultiLabelNERConfig(name="PMLNER", version=datasets.Version("1.0.0"), description="PMLNER dataset"),
    ]

    def _info(self):
        for _, data_path in self.config.data_files.items():
            currnet_data_path = data_path[0] if type(data_path) == datasets.data_files.DataFilesList else data_path
            label_name_path = os.path.join(os.path.dirname(currnet_data_path), _LABEL_FILE)
        label_names = get_label_names(label_name_path)
        logger.critical(f"Read label name from :{label_name_path}, \nlabel_names:{label_names}")
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value(dtype="string"),
                    "labels": datasets.Sequence(datasets.features.ClassLabel(names=label_names)),
                    "unique_id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value(dtype="string")),
                    "label_probs": datasets.Sequence(datasets.Sequence(datasets.Value(dtype="float32"))),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        
        data_files = dict()
        for data_split_type, data_path in self.config.data_files.items():
            data_files[data_split_type] = data_path[0] if type(data_path) == datasets.data_files.DataFilesList else data_path

        generator_list = []
        if "train" in data_files:
            generator_list.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}))
        if "validation" in data_files or "dev" in data_files:
            generator_list.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["validation"]}))
        if "test" in data_files:
            generator_list.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}))
        
        return generator_list

    def _generate_examples(self, filepath):
        logger.info("Generating examples from = %s", filepath)
        file_name = os.path.basename(filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            unique_id = f"{0}-{file_name}"
            tokens = []
            labels = []
            label_probs = []

            for line_idx, line in enumerate(f):
                if (line.strip()=="" or line=="\n") and line_idx!=0:  # For blank line

                    assert len(tokens) == len(labels) ==len(label_probs), \
                        f'guid:{guid}, unique_id:{unique_id}, line_idx: {line_idx},  \
                        len(tokens)({len(tokens)}) != len(labels)({len(labels)}) ({len(label_probs)}) \
                        {" ".join(tokens)} \
                        {" ".join(label_probs)}'

                    if len(tokens) != 0:
                        yield guid, {
                            "id": str(guid),
                            "unique_id": unique_id,
                            "tokens": tokens,
                            "labels": labels,
                            "label_probs": label_probs,
                        }
                        tokens = []
                        labels = []
                        label_probs = []

                        guid += 1
                        unique_id = f"{line_idx+2}-{file_name}" # To make the start of the line as "line_idx"
                    else:
                        logger.critical(f"Two continual empty lines detected! Skip this line. At unique_id:{unique_id}, id:{id}")
                        continue
                else:
                    input_line_parsed = line.splitlines()[0].split("\t")

                    tokens.append(input_line_parsed[0])
                    labels.append(input_line_parsed[-1])
                    label_probs.append(input_line_parsed[1:-1]) # this is a list of lists

            # last example
            yield guid, {
                "id": str(guid),
                "unique_id": unique_id,
                "tokens": tokens,
                "labels": labels,
                "label_probs": label_probs,
            }