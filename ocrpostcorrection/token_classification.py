# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_token_classification.ipynb.

# %% auto 0
__all__ = ['tokenize_and_align_labels_with_tokenizer', 'tokenize_and_align_labels', 'predictions_to_labels',
           'separate_subtoken_predictions', 'merge_subtoken_predictions', 'gather_token_predictions',
           'labels2label_str', 'extract_icdar_output', 'predictions2icdar_output', 'create_perfect_icdar_output']

# %% ../nbs/01_token_classification.ipynb 2
import re
from functools import partial
from collections import defaultdict, Counter

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

# %% ../nbs/01_token_classification.ipynb 5
def tokenize_and_align_labels_with_tokenizer(tokenizer, examples):
    """Tokenize function, to be used as partial with instatiated tokenizer"""
    # Source: https://huggingface.co/docs/transformers/custom_datasets#token-classification-with-wnut-emerging-entities
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:                            # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:              # Only label the first token of a given word.
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# %% ../nbs/01_token_classification.ipynb 6
def tokenize_and_align_labels(tokenizer):
    """Function to tokenize samples and align the labels"""""
    return partial(tokenize_and_align_labels_with_tokenizer, tokenizer)

# %% ../nbs/01_token_classification.ipynb 8
def predictions_to_labels(predictions):
    return np.argmax(predictions, axis=2)

# %% ../nbs/01_token_classification.ipynb 11
def separate_subtoken_predictions(word_ids, preds):
    #print(len(word_ids), word_ids)
    result = defaultdict(list)
    for word_idx, p_label in zip(word_ids, preds):
        #print(word_idx, p_label)
        if word_idx is not None:
            result[word_idx].append(p_label)
    return dict(result)


# %% ../nbs/01_token_classification.ipynb 13
def merge_subtoken_predictions(subtoken_predictions):
    token_level_predictions = []
    for word_idx, preds in subtoken_predictions.items():
        token_label = 0
        c = Counter(preds)
        #print(c)
        if c[1] > 0 and c[1] >= c[2]:
            token_label = 1
        elif c[2] > 0 and c[2] >= c[1]:
            token_label = 2

        token_level_predictions.append(token_label)
    return token_level_predictions

# %% ../nbs/01_token_classification.ipynb 15
def gather_token_predictions(preds):
    """Gather potentially overlapping token predictions"""
    labels = defaultdict(list)
        
    #print(len(text.input_tokens))
    #print(preds)
    for start, lbls in preds.items():
        for i, label in enumerate(lbls):
            labels[int(start)+i].append(label)
    #print('LABELS')
    #print(labels)
    return dict(labels)

# %% ../nbs/01_token_classification.ipynb 17
def labels2label_str(labels):
    label_str = []

    for i, token in enumerate(labels):
        #print(i, token, labels[i])
        if 2 in labels[i]:
            label_str.append('2')
        elif 1 in labels[i]:
            label_str.append('1')
        else:
            label_str.append('0')
    label_str = ''.join(label_str)
    return label_str

# %% ../nbs/01_token_classification.ipynb 19
def extract_icdar_output(label_str, input_tokens):
    #print(label_str, input_tokens)
    #print(len(label_str), len(input_tokens))
    text_output = {}

    # Correct use of 2 (always following a 1)
    regex = r'12*'

    for match in re.finditer(regex, label_str):
        #print(match)
        #print(match.group())
        num_tokens = len(match.group())
        #print(match.start(), len(input_tokens))
        idx = input_tokens[match.start()].start
        text_output[f'{idx}:{num_tokens}'] = {}

    # Incorrect use of 2 (following a 0) -> interpret first 2 as 1
    regex = r'02+'

    for match in re.finditer(regex, label_str):
        #print(match)
        #print(match.group())
        num_tokens = len(match.group()) - 1
        idx = input_tokens[match.start()+1].start
        text_output[f'{idx}:{num_tokens}'] = {}
    
    return text_output

# %% ../nbs/01_token_classification.ipynb 25
def predictions2icdar_output(samples, predictions, tokenizer, data_test):
    """Convert predictions into icdar output format"""
    #print('samples', len(samples))
    #print(samples)
    #print(samples[0].keys())
    #for sample in samples:
    #    print(sample.keys()) 

    tokenized_samples = tokenizer(samples["tokens"], truncation=True, is_split_into_words=True)
    #print(samples)

    #for sample in samples:
    #    print(sample.keys())
    
    # convert predictions to labels (label_ids)
    #p = np.argmax(predictions, axis=2)
    #print(p)

    converted = defaultdict(dict)

    for i, (sample, preds) in enumerate(zip(samples, predictions)):
        #print(sample.keys())
        #label = sample['tags']
        #print(label)
        #print(len(preds), preds)
        word_ids = tokenized_samples.word_ids(batch_index=i)  # Map tokens to their respective word.
        result = separate_subtoken_predictions(word_ids, preds)
        new_tags = merge_subtoken_predictions(result)

        #print('pred', len(new_tags), new_tags)
        #print('tags', len(label), label)
        
        #print(sample)
        #print(sample['key'], sample['start_token_id'])
        converted[sample['key']][sample['start_token_id']] = new_tags
    
    output = {}
    for key, preds in converted.items():
        labels = defaultdict(list)
        #print(key)
        labels = gather_token_predictions(preds)
        label_str = labels2label_str(labels)
        try:
            text = data_test[key]
        except KeyError:
            logger.warning(f'No data found for text {key}')
        output[key] = extract_icdar_output(label_str, text.input_tokens)

    return output

# %% ../nbs/01_token_classification.ipynb 27
def create_perfect_icdar_output(data):
    output = {}
    for key, text_obj in data.items():
        label_str = ''.join([str(t.label) for t in text_obj.input_tokens])
        output[key] = extract_icdar_output(label_str, data[key].input_tokens)
    return output
