import datasets
import model_handling
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, Any
from transformers.file_utils import PaddingStrategy
import re
import os
from tqdm import tqdm
import time
import json
import random
import regtag
from dataclasses import dataclass
import validators

import utils

regexp = re.compile(r"\d{4}[\-/]\d{2}[\-/]\d{2}t\d{2}:\d{2}:\d{2}")
target_bias_words = set(regtag.get_general_en_word())
tokenizer = None


def get_bias_words():
    regtag.augment.get_random_oov()
    return list(regtag.augment.oov_dict.keys())


def check_common_phrase(word):
    if validators.email(word.replace(' @', '@')):
        return True
    if validators.domain(word):
        return True
    if validators.url(word):
        return True
    if word in regtag.get_general_en_word():
        return True
    return False


@dataclass
class DataCollatorForNormSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def bias_phrases_extractor(self, features, max_bias_per_sample=15):
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
        phrase_candidate_revised = []
        phrase_candidate_common = []
        raw_phrase_candidate = []
        for item in phrase_candidate:
            raw_item = self.tokenizer.sp_model.DecodePieces(item.split())
            if check_common_phrase(raw_item):
                phrase_candidate_common.append(raw_item)
            else:
                phrase_candidate_revised.append(item)
            raw_phrase_candidate.append(raw_item)

        remain_phrase = max(0, max_bias_per_sample * len(features) - len(phrase_candidate_revised))

        if remain_phrase > 0:
            words_candidate = list(
                set(get_bias_words()) - set(raw_phrase_candidate))
            random.shuffle(words_candidate)
            phrase_candidate_revised += [' '.join(self.tokenizer.sp_model.EncodeAsPieces(item)[:5]) for item in
                                         words_candidate[:remain_phrase]]

        for i in range(len(features)):
            sample_bias_lables = []
            for w_idx, w in enumerate(sample_output_words[i]):
                try:
                    sample_bias_lables.extend(
                        [phrase_candidate_revised.index(w) + 1] * features[i]['outputs_length'][w_idx])
                except:
                    # random ignore 0 label
                    if random.random() < 0.5:
                        sample_bias_lables.extend([0] * features[i]['outputs_length'][w_idx])
                    else:
                        sample_bias_lables.extend([self.label_pad_token_id] * features[i]['outputs_length'][w_idx])
            bias_labels.append(sample_bias_lables)
            assert len(sample_bias_lables) == len(features[i]['outputs']), "{} vs {}".format(sample_bias_lables,
                                                                                             features[i]['outputs'])

        # phrase_candidate_ids = [self.tokenizer.encode(item) for item in phrase_candidate]
        phrase_candidate_ids = [self.tokenizer.encode(self.tokenizer.sp_model.DecodePieces(item.split())) for item in
                                phrase_candidate_revised]
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
        start_time = time.time()
        batch_src, batch_tgt = [], []
        for item in features:
            src_spans, tgt_spans = utils.make_spoken(item['text'])
            batch_src.append(src_spans)
            batch_tgt.append(tgt_spans)
        print("Make src-tgt {}s".format(time.time() - start_time))
        start_time = time.time()

        features = preprocess_function({"src": batch_src, "tgt": batch_tgt})


        print("Make feature {}s".format(time.time() - start_time))
        start_time = time.time()

        phrase_candidate_ids, phrase_candidate_mask, samples_bias_labels = self.bias_phrases_extractor(features)
        # print("Make bias {}s".format(time.time() - start_time))
        # start_time = time.time()

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["outputs"] for feature in features] if "outputs" in features[0].keys() else None
        spoken_labels = [feature["spoken_label"] for feature in features] if "spoken_label" in features[0].keys() else None
        spoken_idx = [feature["src_spoken_idx"] for feature in features] if "src_spoken_idx" in features[0].keys() else None

        word_src_lengths = [feature["inputs_length"] for feature in features] if "inputs_length" in features[0].keys() else None
        word_tgt_lengths = [feature["outputs_length"] for feature in features] if "outputs_length" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_src_length = max(len(l) for l in spoken_labels)
            max_spoken_idx_length = max(len(l) for l in spoken_idx)
            max_word_src_length = max(len(l) for l in word_src_lengths)
            max_word_tgt_length = max(len(l) for l in word_tgt_lengths)

            padding_side = self.tokenizer.padding_side
            for feature, bias_labels in zip(features, samples_bias_labels):
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["outputs"]))
                remainder_word_tgt_length = [0] * (max_word_tgt_length - len(feature["outputs_length"]))
                remainder_spoken = [self.label_pad_token_id] * (max_src_length - len(feature["spoken_label"]))
                remainder_spoken_idx = [self.label_pad_token_id] * (max_spoken_idx_length - len(feature["src_spoken_idx"]))
                remainder_word_src_length = [0] * (max_word_src_length - len(feature["inputs_length"]))

                feature["labels"] = (
                    feature["outputs"] + [
                        self.tokenizer.eos_token_id] + remainder if padding_side == "right" else remainder + feature[
                        "outputs"] + [self.tokenizer.eos_token_id]
                )
                feature["labels_bias"] = (
                    bias_labels + [0] + remainder if padding_side == "right" else remainder + bias_labels + [0]
                )

                feature["spoken_label"] = [self.label_pad_token_id] + feature["spoken_label"] + [self.label_pad_token_id]
                feature["spoken_label"] = feature["spoken_label"] + remainder_spoken if padding_side == "right" else remainder_spoken + feature["spoken_label"]
                feature["src_spoken_idx"] = feature["src_spoken_idx"] + remainder_spoken_idx

                feature['inputs_length'] = [1] + feature['inputs_length'] + [1]
                feature['outputs_length'] = feature['outputs_length'] + [1]

                feature["inputs_length"] = feature["inputs_length"] + remainder_word_src_length
                feature["outputs_length"] = feature["outputs_length"] + remainder_word_tgt_length


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
        spoken_label = self.tokenizer.pad({"input_ids": [feature["spoken_label"] for feature in features]},
                                          return_tensors=return_tensors)['input_ids']
        spoken_idx = self.tokenizer.pad({"input_ids": [feature["src_spoken_idx"] for feature in features]},
                                        return_tensors=return_tensors)['input_ids'] + 1  # 1 for bos token
        word_src_lengths = self.tokenizer.pad({"input_ids": [feature["inputs_length"] for feature in features]},
                                              return_tensors=return_tensors)['input_ids']
        word_tgt_lengths = self.tokenizer.pad({"input_ids": [feature["outputs_length"] for feature in features]},
                                              return_tensors=return_tensors)['input_ids']

        features = {
            "input_ids": features_inputs["input_ids"],
            "spoken_label": spoken_label,
            "spoken_idx": spoken_idx,
            "word_src_lengths": word_src_lengths,
            "word_tgt_lengths": word_tgt_lengths,
            "attention_mask": features_inputs["attention_mask"],
            "bias_input_ids": bias_phrases_inputs["input_ids"],
            "bias_attention_mask": bias_phrases_inputs["attention_mask"],
            "labels": outputs,
            "labels_bias": outputs_bias
        }

        print("Make batch {}s".format(time.time() - start_time))
        start_time = time.time()

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


# data init
def init_data(train_corpus_path='./data-bin/raw/train.txt',
              test_corpus_path='./data-bin/raw/valid.txt'):
    dataset_oov = datasets.load_dataset('text', data_files={"train": train_corpus_path,
                                                            "test": test_corpus_path})

    print(dataset_oov)
    return dataset_oov


def preprocess_function(batch):

    global tokenizer
    if tokenizer is None:
        tokenizer = model_handling.init_tokenizer()

    features = []
    for src_words, tgt_words in zip(batch["src"], batch["tgt"]):
        src_ids, pad_ids, src_lengths, tgt_ids, tgt_lengths = [], [], [], [], []
        src_spoken_label = []  # 0: "O", 1: "B", 2: "I"

        src_spoken_idx = []
        tgt_spoken_ids = []

        for idx, (src, tgt) in enumerate(zip(src_words, tgt_words)):
            is_remain = False
            if src == tgt:
                is_remain = True

            src_tokenized = tokenizer(src)
            if len(src_tokenized['input_ids']) < 3:
                continue
            # hardcode fix tokenizer email
            if validators.email(tgt):
                tgt_tokenized = tokenizer(tgt.replace('@', ' @'))
            else:
                tgt_tokenized = tokenizer(tgt)
            if len(tgt_tokenized['input_ids']) < 3:
                continue
            src_ids.extend(src_tokenized["input_ids"][1:-1])
            if is_remain:
                src_spoken_label.extend([0 if random.random() < 0.5 else -100 for _ in range(len(src_tokenized["input_ids"][1:-1]))])
                if random.random() < 0.1:
                    # Random pick normal word for spoken norm
                    src_spoken_idx.append(idx)
                    tgt_spoken_ids.append(tgt_tokenized["input_ids"][1:-1])
            else:
                src_spoken_label.extend([1] + [2] * (len(src_tokenized["input_ids"][1:-1]) - 1))
                src_spoken_idx.append(idx)
                tgt_spoken_ids.append(tgt_tokenized["input_ids"][1:-1])

            pad_ids.extend(src_tokenized["attention_mask"][1:-1])
            src_lengths.append(len(src_tokenized["input_ids"]) - 2)
            tgt_ids.extend(tgt_tokenized["input_ids"][1:-1])
            tgt_lengths.append(len(tgt_tokenized["input_ids"]) - 2)
            if len(src_ids) > 80 or len(tgt_ids) > 80:
                # print("Ignore sample")
                break

        if len(src_ids) < 1 or len(tgt_ids) < 1:
            continue
        if len(src_ids) < 2:
            print(src_words, tgt_words)
        # print(len(src_ids), len(tgt_ids))

        features.append({
            "input_ids": src_ids,
            "attention_mask": pad_ids,
            "spoken_label": src_spoken_label,
            "inputs_length": src_lengths,
            "outputs": tgt_ids,
            "outputs_length": tgt_lengths,
            "src_spoken_idx": src_spoken_idx,
            "tgt_spoken_ids": tgt_spoken_ids
        })

    return features


if __name__ == "__main__":
    split_datasets = init_data()

    model, model_tokenizer = model_handling.init_model()
    data_collator = DataCollatorForNormSeq2Seq(model_tokenizer, model=model)
    import time
    start = time.time()
    batch = data_collator([split_datasets["train"][i] for i in [random.randint(0, 900) for _ in range(0, 64)]])
    # print(batch)
    print("{}s".format(time.time() - start))
