import datasets
import model_handling
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, Any
from transformers.file_utils import PaddingStrategy
import re
import os
from tqdm import tqdm
import json
import random
import numpy as np
from dataclasses import dataclass

regexp = re.compile(r"\d{4}[\-/]\d{2}[\-/]\d{2}t\d{2}:\d{2}:\d{2}")
tokenizer = None


@dataclass
class DataCollatorForNormSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def bias_phrases_extractor(self, features, max_bias_per_sample=3):
        # src_ids, src_length, tgt_ids, tgt_length
        phrase_candidate = []
        sample_output_words = []
        bias_labels = []

        for sample in features:
            words = []
            for idx, (src_word_len, tgt_word_len) in enumerate(zip(sample['inputs_length'], sample['outputs_length'])):
                src_start_idx = sum(sample['inputs_length'][:idx])
                tgt_start_idx = sum(sample['outputs_length'][:idx])
                word_input = self.tokenizer.decode(sample['input_ids'][src_start_idx: src_start_idx + src_word_len])
                word_output = self.tokenizer.decode(sample['outputs'][tgt_start_idx: tgt_start_idx + tgt_word_len])
                words.append(word_output)
                if word_input != word_output and not any(map(str.isdigit, word_output)):
                    phrase_candidate.append(word_output)
            sample_output_words.append(words)

        phrase_candidate = list(set(phrase_candidate))
        # random.shuffle(phrase_candidate)
        # phrase_candidate = phrase_candidate[:int(len(phrase_candidate) * 0.6)]

        remain_phrase = max(0, max_bias_per_sample * len(features) - len(phrase_candidate))
        if remain_phrase > 0:
            words_candidate = list(
                set([item for sublist in sample_output_words for item in sublist]) - set(phrase_candidate))
            phrase_candidate += random.choices(words_candidate, k=remain_phrase)

        for i in range(len(features)):
            sample_bias_lables = []
            for w_idx, w in enumerate(sample_output_words[i]):
                try:
                    sample_bias_lables.extend([phrase_candidate.index(w) + 1] * features[i]['outputs_length'][w_idx])
                except:
                    # random ignore 0 label
                    if random.random() < 0.5:
                        sample_bias_lables.extend([0] * features[i]['outputs_length'][w_idx])
                    else:
                        sample_bias_lables.extend([self.label_pad_token_id] * features[i]['outputs_length'][w_idx])
            bias_labels.append(sample_bias_lables)
            assert len(sample_bias_lables) == len(features[i]['outputs'])

        # phrase_candidate_ids = [self.tokenizer.encode(item) for item in phrase_candidate]
        phrase_candidate_ids = [self.tokenizer.encode(self.tokenizer.sp_model.DecodePieces(item.split())) for item in
                                phrase_candidate]
        phrase_candidate_mask = [[self.tokenizer.pad_token_id] * len(item) for item in phrase_candidate_ids]

        return phrase_candidate_ids, phrase_candidate_mask, bias_labels
        # pass

    def encode_list_string(self, list_text):
        text_tokenized = self.tokenizer(list_text)
        return self.tokenizer.pad(
            text_tokenized,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

    def __call__(self, features, return_tensors=None):
        phrase_candidate_ids, phrase_candidate_mask, samples_bias_labels = self.bias_phrases_extractor(features)

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["outputs"] for feature in features] if "outputs" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature, bias_labels in zip(features, samples_bias_labels):
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["outputs"]))

                feature["labels"] = (
                    feature["outputs"] + [
                        self.tokenizer.eos_token_id] + remainder if padding_side == "right" else remainder + feature[
                        "outputs"] + [self.tokenizer.eos_token_id]
                )
                feature["labels_bias"] = (
                    bias_labels + [0] + remainder if padding_side == "right" else remainder + bias_labels + [0]
                )

        # padding input
        features_inputs = [{
            "input_ids": [self.tokenizer.bos_token_id] + item["input_ids"] + [self.tokenizer.eos_token_id],
            "attention_mask": [self.tokenizer.pad_token_id] + item["attention_mask"] + [self.tokenizer.pad_token_id]
        } for item in features]
        features_inputs = self.tokenizer.pad(
            features_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        bias_phrases_inputs = [{
            "input_ids": ids,
            "attention_mask": mask
        } for ids, mask in zip(phrase_candidate_ids, phrase_candidate_mask)]
        bias_phrases_inputs = self.tokenizer.pad(
            bias_phrases_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        outputs = self.tokenizer.pad({"input_ids": [feature["labels"] for feature in features]},
                                     return_tensors=return_tensors)['input_ids']
        outputs_bias = self.tokenizer.pad({"input_ids": [feature["labels_bias"] for feature in features]},
                                          return_tensors=return_tensors)['input_ids']

        features = {
            "input_ids": features_inputs["input_ids"],
            "attention_mask": features_inputs["attention_mask"],
            "bias_input_ids": bias_phrases_inputs["input_ids"],
            "bias_attention_mask": bias_phrases_inputs["attention_mask"],
            "labels": outputs,
            "labels_bias": outputs_bias
        }

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def ignore_pattern(word):
    if regexp.search(word):
        return True
    return False


def format_sample(raw_lines):
    count_change = 0
    try:
        src = []
        tgt = []
        for line in raw_lines:
            rows = line.strip().split('\t')
            src_word = rows[0].lower()
            tgt_word = rows[1].lower()
            if len(tgt_word) == 0:
                continue
            if ignore_pattern(tgt_word):
                count_change = 0
                break
            if src_word != tgt_word:
                count_change += 1
            if src_word not in [',', '.', '?', '!', ':', '-']:
                src.append(src_word)
                tgt.append(tgt_word)
    except:
        return None, None
    if count_change > 0:
        return src, tgt
    else:
        return None, None


# data init
def init_data():
    train_dataset_path = './data-bin/train.json'
    test_dataset_path = './data-bin/valid.json'
    max_sample = 1000
    if not (os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path)):
        samples = []
        with open('./data-bin/data_augment_v2.src-tgt', 'r', encoding='utf-8') as file:
            sample = []
            for line in tqdm(file):
                if len(line.strip()) == 0:
                    if len(sample) > 0:
                        inputs, ouputs = format_sample(sample)
                        if inputs is not None:
                            samples.append((inputs, ouputs))
                        sample = []
                        # if max_sample < len(samples):
                        #     break
                else:
                    sample.append(line)
            if len(sample) > 0:
                inputs, ouputs = format_sample(sample)
                if inputs is not None:
                    samples.append((inputs, ouputs))

        # random.shuffle(samples)

        train_data = samples[:int(len(samples) * 0.98)]
        valid_data = samples[int(len(samples) * 0.98):]

        with open(train_dataset_path, 'w', encoding='utf-8') as file:
            for item in tqdm(train_data):
                file.write(
                    "{}\n".format(json.dumps({"src": item[0], "tgt": item[1]}, ensure_ascii=False)))
        with open(test_dataset_path, 'w', encoding='utf-8') as file:
            for item in tqdm(valid_data):
                file.write(
                    "{}\n".format(json.dumps({"src": item[0], "tgt": item[1]}, ensure_ascii=False)))

    dataset_oov = datasets.load_dataset('json', data_files={"train": train_dataset_path,
                                                            "test": test_dataset_path})
    print(dataset_oov)
    return dataset_oov


def preprocess_function(batch):
    global tokenizer
    if tokenizer is None:
        tokenizer = model_handling.init_tokenizer()
    inputs = []
    inputs_length = []
    attention_mask = []
    outputs = []
    outputs_length = []
    for src_words, tgt_words in zip(batch["src"], batch["tgt"]):
        src_ids, pad_ids, src_lengths, tgt_ids, tgt_lengths = [], [], [], [], []
        for src, tgt in zip(src_words, tgt_words):
            src_tokenized = tokenizer(src)
            tgt_tokenized = tokenizer(tgt)
            src_ids.extend(src_tokenized["input_ids"][1:-1])
            pad_ids.extend(src_tokenized["attention_mask"][1:-1])
            src_lengths.append(len(src_tokenized["input_ids"]) - 2)
            tgt_ids.extend(tgt_tokenized["input_ids"][1:-1])
            tgt_lengths.append(len(tgt_tokenized["input_ids"]) - 2)
        if len(src_ids) > 500 or len(tgt_ids) > 500:
            # print("Ignore sample")
            continue
        inputs.append(src_ids)
        inputs_length.append(src_lengths)
        attention_mask.append(pad_ids)
        outputs.append(tgt_ids)
        outputs_length.append(tgt_lengths)

    batch["input_ids"] = inputs
    batch["attention_mask"] = attention_mask
    batch["inputs_length"] = inputs_length

    batch["outputs"] = outputs
    batch["outputs_length"] = outputs_length

    return batch


if __name__ == "__main__":
    split_datasets = init_data()

    #
    tokenized_datasets = split_datasets.map(
        preprocess_function,
        batched=True,
        batch_size=4,
        num_proc=4,
        remove_columns=split_datasets["train"].column_names,
        cache_file_names={"train": "./cache/train_datasets.arrow", "test": "./cache/test_datasets.arrow"}
    )
    model, model_tokenizer = model_handling.init_model()
    data_collator = DataCollatorForNormSeq2Seq(model_tokenizer, model=model)
    batch = data_collator([tokenized_datasets["train"][i] for i in range(0, 3)])
    print(batch)
