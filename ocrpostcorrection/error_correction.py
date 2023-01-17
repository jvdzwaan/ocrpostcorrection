# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_error_correction.ipynb.

# %% auto 0
__all__ = ['UNK_IDX', 'PAD_IDX', 'BOS_IDX', 'EOS_IDX', 'special_symbols', 'get_tokens_with_OCR_mistakes', 'yield_tokens',
           'generate_vocabs', 'SimpleCorrectionDataset', 'sequential_transforms', 'tensor_transform',
           'get_text_transform', 'collate_fn_with_text_transform', 'collate_fn', 'EncoderRNN', 'AttnDecoderRNN',
           'SimpleCorrectionSeq2seq', 'validate_model', 'GreedySearchDecoder', 'indices2string',
           'predict_and_convert_to_str']

# %% ../nbs/02_error_correction.ipynb 2
import dataclasses
import random

from functools import partial
from itertools import chain
from typing import Iterable, List, Dict

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from .icdar_data import Text, normalized_ed

# %% ../nbs/02_error_correction.ipynb 5
def get_tokens_with_OCR_mistakes(data: Dict[str, Text], 
                                data_test: Dict[str, Text], 
                                val_files: List[str]) \
                                    -> pd.DataFrame:
    """Return pandas dataframe with all OCR mistakes from train, val, and test"""
    tokens = []
    # Train and val
    for key, d in data.items():
        for token in d.tokens:
            if token.ocr.strip() != token.gs.strip():
                r = dataclasses.asdict(token)
                r['language'] = key[:2]
                r['subset'] = key.split('/')[1]

                if key in val_files:
                    r['dataset'] = 'val'
                else:
                    r['dataset'] = 'train'

                tokens.append(r)
    # Test
    for key, d in data_test.items():
        for token in d.tokens:
            if token.ocr.strip() != token.gs.strip():
                r = dataclasses.asdict(token)
                r['language'] = key[:2]
                r['subset'] = key.split('/')[1]
                r['dataset'] = 'test'

                tokens.append(r)
    tdata = pd.DataFrame(tokens)
    tdata = _add_update_data_properties(tdata)

    return tdata

def _add_update_data_properties(tdata: pd.DataFrame) -> pd.DataFrame:
    """Add and update data properties for calculating statistics"""
    tdata['ocr'] = tdata['ocr'].apply(lambda x: x.strip())
    tdata['gs'] = tdata['gs'].apply(lambda x: x.strip())
    tdata['len_ocr'] = tdata.apply(lambda row: len(row.ocr), axis=1)
    tdata['len_gs'] = tdata.apply(lambda row: len(row.gs), axis=1)
    tdata['diff'] = tdata.len_ocr - tdata.len_gs
    return tdata


# %% ../nbs/02_error_correction.ipynb 10
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# %% ../nbs/02_error_correction.ipynb 11
def yield_tokens(data, col):
    """Helper function to create vocabulary containing characters"""
    for token in data[col].to_list():
        for char in token:
            yield char

# %% ../nbs/02_error_correction.ipynb 12
def generate_vocabs(train):
    """Generate ocr and gs vocabularies from the train set"""
    vocab_transform = {}
    for name in ('ocr', 'gs'):
        vocab_transform[name] = build_vocab_from_iterator(yield_tokens(train, name),
                                                          min_freq=1,
                                                          specials=special_symbols,
                                                          special_first=True)
    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for name in ('ocr', 'gs'):
        vocab_transform[name].set_default_index(UNK_IDX)
    
    return vocab_transform

# %% ../nbs/02_error_correction.ipynb 16
class SimpleCorrectionDataset(Dataset):
    def __init__(self, data, max_len=10):
        self.ds = data.query(f'len_ocr <= {max_len}').query(f'len_gs <= {max_len}').copy()
        self.ds = self.ds.reset_index(drop=False)

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, idx):
        sample = self.ds.loc[idx]

        return [char for char in sample.ocr], [char for char in sample.gs]

# %% ../nbs/02_error_correction.ipynb 24
def sequential_transforms(*transforms):
    """Helper function to club together sequential operations"""
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    """Function to add BOS/EOS and create tensor for input sequence indices"""
    return torch.cat((torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def get_text_transform(vocab_transform):
    """Returns text transforms to convert raw strings into tensors indices"""
    text_transform = {}
    for name in ('ocr', 'gs'):
        text_transform[name] = sequential_transforms(vocab_transform[name],  # Numericalization (char -> idx)
                                                     tensor_transform) # Add BOS/EOS and create tensor
    return text_transform

# %% ../nbs/02_error_correction.ipynb 27
def collate_fn_with_text_transform(text_transform, batch):
    """Function to collate data samples into batch tensors, to be used as partial with instatiated text_transform"""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform['ocr'](src_sample))
        tgt_batch.append(text_transform['gs'](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch.to(torch.int64), tgt_batch.to(torch.int64)


def collate_fn(text_transform):
    """Function to collate data samples into batch tensors"""
    return partial(collate_fn_with_text_transform, text_transform)

# %% ../nbs/02_error_correction.ipynb 31
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # print('Encoder')
        # print('input size', input.size())
        # print('hidden size', hidden.size())
        embedded = self.embedding(input) 
        # print('embedded size', embedded.size())
        # print(embedded)
        # print('embedded size met view', embedded.view(1, 1, -1).size())
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

# %% ../nbs/02_error_correction.ipynb 32
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=11):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # print('embedded size', embedded.size())
        # print(embedded)
        embedded = torch.permute(embedded, (1, 0, 2))
        # print('permuted embedded size', embedded.size())
        # print(embedded)

        # print('hidden size', hidden.size())
        # print(hidden)

        # print('permuted embedded[0] size', embedded[0].size())
        # print('hidden[0] size', hidden[0].size())

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # print('attn_weights', attn_weights.size())
        # print('attn_weights unsqueeze(1)', attn_weights.unsqueeze(1).size())
        # print('encoder outputs', encoder_outputs.size())


        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        # print('embedded[0]', embedded[0].size())
        # print('attn_applied', attn_applied.size())
        # print('attn_applied squeeze', attn_applied.squeeze().size())
        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        # print('output', output.size())
        output = self.attn_combine(output).unsqueeze(0)
        # print('output', output.size())

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        # print(f'output: {output.size()}; hidden: {hidden.size()}; attn_weigts: {attn_weights.size()}')

        return output, hidden, attn_weights

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

# %% ../nbs/02_error_correction.ipynb 33
class SimpleCorrectionSeq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, max_length, 
                 teacher_forcing_ratio, device='cpu'):
        super(SimpleCorrectionSeq2seq, self).__init__()

        self.device = device

        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.max_length = max_length+1

        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size, 
                                      dropout_p=dropout, max_length=self.max_length)
    
    def forward(self, input, encoder_hidden, target):
        # input is src seq len x batch size
        # input voor de encoder (1 stap) moet zijn input seq len x batch size x 1
        input_tensor = input.unsqueeze(2)
        # print('input tensor size', input_tensor.size())

        input_length = input.size(0)

        batch_size = input.size(1)

        # Encoder part
        encoder_outputs = torch.zeros(batch_size, self.max_length, self.encoder.hidden_size, 
                                      device=self.device)
        # print('encoder outputs size', encoder_outputs.size())
    
        for ei in range(input_length):
            # print(f'Index {ei}; input size: {input_tensor[ei].size()}; encoder hidden size: {encoder_hidden.size()}')
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            # print('Index', ei)
            # print('encoder output size', encoder_output.size())
            # print('encoder outputs size', encoder_outputs.size())
            # print('output selection size', encoder_output[:, 0].size())
            # print('ouput to save', encoder_outputs[:,ei].size())
            encoder_outputs[:, ei] = encoder_output[0, 0]
        
        # print('encoder outputs', encoder_outputs)
        # print('encoder hidden', encoder_hidden)

        # Decoder part
        # Target = seq len x batch size
        # Decoder input moet zijn: batch_size x 1 (van het eerste token = BOS)
        target_length = target.size(0)

        decoder_input = torch.tensor([[BOS_IDX] for _ in range(batch_size)], 
                                     device=self.device)
        # print('decoder input size', decoder_input.size())

        decoder_outputs = torch.zeros(batch_size, self.max_length, self.decoder.output_size, 
                                      device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target[di, :].unsqueeze(1)  # Teacher forcing
                #print('decoder input size:', decoder_input.size())
            else:
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input
                #print('decoder input size:', decoder_input.size())

            # print(f'Index {di}; decoder output size: {decoder_output.size()}; decoder input size: {decoder_input.size()}')
            decoder_outputs[:, di] = decoder_output

        # Zero out probabilities for padded chars
        target_masks = (target != PAD_IDX).float()

        # Compute log probability of generating true target words
        # print('P (decoder_outputs)', decoder_outputs.size())
        # print(target.transpose(0, 1))
        # print('Index', target.size(), target.transpose(0, 1).unsqueeze(-1))
        target_gold_std_log_prob = torch.gather(decoder_outputs, index=target.transpose(0, 1).unsqueeze(-1), dim=-1).squeeze(-1) * target_masks.transpose(0, 1)
        #print(target_gold_std_log_prob)
        scores = target_gold_std_log_prob.sum(dim=1)

        #print(scores)

        return scores, encoder_outputs


# %% ../nbs/02_error_correction.ipynb 36
def validate_model(model, dataloader, device):
    cum_loss = 0
    cum_examples = 0

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            batch_size = src.size(1)

            encoder_hidden = model.encoder.initHidden(batch_size=batch_size, device=device)

            example_losses, decoder_ouputs = model(src, encoder_hidden, tgt)
            example_losses = -example_losses
            batch_loss = example_losses.sum()

            bl = batch_loss.item()
            cum_loss += bl
            cum_examples += batch_size

    if was_training:
        model.train()

    return cum_loss/cum_examples

# %% ../nbs/02_error_correction.ipynb 44
class GreedySearchDecoder(nn.Module):
    def __init__(self, model):
        super(GreedySearchDecoder, self).__init__()
        self.max_length = model.max_length
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.device = model.device

    def forward(self, input, target):
        # input is src seq len x batch size
        # input voor de encoder (1 stap) moet zijn input seq len x batch size x 1
        input_tensor = input.unsqueeze(2)
        # print('input tensor size', input_tensor.size())

        input_length = input.size(0)

        batch_size = input.size(1)
        encoder_hidden = self.encoder.initHidden(batch_size, self.device)

        # Encoder part    
        encoder_outputs = torch.zeros(batch_size, self.max_length, self.encoder.hidden_size, 
                                      device=self.device)
        # print('encoder outputs size', encoder_outputs.size())
    
        for ei in range(input_length):
            # print(f'Index {ei}; input size: {input_tensor[ei].size()}; encoder hidden size: {encoder_hidden.size()}')
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            # print('Index', ei)
            # print('encoder output size', encoder_output.size())
            # print('encoder outputs size', encoder_outputs.size())
            # print('output selection size', encoder_output[:, 0].size())
            # print('ouput to save', encoder_outputs[:,ei].size())
            encoder_outputs[:, ei] = encoder_output[0, 0]
        
        # print('encoder outputs', encoder_outputs)
        # print('encoder hidden', encoder_hidden)

        # Decoder part
        # Target = seq len x batch size
        # Decoder input moet zijn: batch_size x 1 (van het eerste token = BOS)
        target_length = target.size(0)

        decoder_input = torch.tensor([[BOS_IDX] for _ in range(batch_size)], 
                                     device=self.device)
        # print('decoder input size', decoder_input.size())

        all_tokens = torch.zeros(batch_size, self.max_length, device=self.device, dtype=torch.long)
        # print('all_tokens size', all_tokens.size())
        decoder_hidden = encoder_hidden
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            # print('decoder input size:', decoder_input.size())
            # print('decoder input squeezed', decoder_input.clone().squeeze())

            # Record token
            all_tokens[:, di] = decoder_input.clone().squeeze(1)
            # print('all_tokens', all_tokens)

        return all_tokens

# %% ../nbs/02_error_correction.ipynb 46
def indices2string(indices, itos):
    output = []
    for idxs in indices:
        #print(idxs)
        string = []
        for idx in idxs:
            if idx not in (UNK_IDX, PAD_IDX, BOS_IDX):
                if idx == EOS_IDX:
                    break
                else:
                    string.append(itos[idx])
        word = ''.join(string)
        output.append(word)
    return output

# %% ../nbs/02_error_correction.ipynb 48
def predict_and_convert_to_str(model, dataloader, tgt_vocab, device):
    was_training = model.training
    model.eval()

    decoder = GreedySearchDecoder(model)

    itos = tgt_vocab.get_itos()
    output_strings = []

    with torch.no_grad():
        for src, tgt in tqdm(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            predicted_indices = decoder(src, tgt)
            
            strings_batch = indices2string(predicted_indices, itos)
            for s in strings_batch:
                output_strings.append(s)

    if was_training:
        model.train()

    return output_strings
