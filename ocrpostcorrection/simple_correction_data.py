# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02a_simple_correction_data.ipynb.

# %% auto 0
__all__ = ['SimpleCorrectionDataset', 'collate_fn_with_text_transform', 'collate_fn', 'validate_model', 'train_model',
           'GreedySearchDecoder', 'predict_and_convert_to_str']

# %% ../nbs/02a_simple_correction_data.ipynb 2
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from ocrpostcorrection.error_correction import (
    BOS_IDX,
    PAD_IDX,
    generate_vocabs,
    get_text_transform,
    get_tokens_with_OCR_mistakes,
    indices2string,
)

# %% ../nbs/02a_simple_correction_data.ipynb 6
class SimpleCorrectionDataset(Dataset):
    def __init__(self, data, max_len=10):
        self.ds = (
            data.query(f"len_ocr <= {max_len}").query(f"len_gs <= {max_len}").copy()
        )
        self.ds = self.ds.reset_index(drop=False)

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, idx):
        sample = self.ds.loc[idx]

        return [char for char in sample.ocr], [char for char in sample.gs]

# %% ../nbs/02a_simple_correction_data.ipynb 11
def collate_fn_with_text_transform(text_transform, batch):
    """Function to collate data samples into batch tensors, to be used as partial with instatiated text_transform"""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform["ocr"](src_sample))
        tgt_batch.append(text_transform["gs"](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch.to(torch.int64), tgt_batch.to(torch.int64)


def collate_fn(text_transform):
    """Function to collate data samples into batch tensors"""
    return partial(collate_fn_with_text_transform, text_transform)

# %% ../nbs/02a_simple_correction_data.ipynb 16
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

            encoder_hidden = model.encoder.initHidden(
                batch_size=batch_size, device=device
            )

            example_losses, decoder_ouputs = model(src, encoder_hidden, tgt)
            example_losses = -example_losses
            batch_loss = example_losses.sum()

            bl = batch_loss.item()
            cum_loss += bl
            cum_examples += batch_size

    if was_training:
        model.train()

    return cum_loss / cum_examples

# %% ../nbs/02a_simple_correction_data.ipynb 19
def train_model(
    train_dl,
    val_dl,
    model=None,
    optimizer=None,
    num_epochs=5,
    valid_niter=5000,
    model_save_path="model.rar",
    max_num_patience=5,
    max_num_trial=5,
    lr_decay=0.5,
    device="cpu",
):
    num_iter = 0
    report_loss = 0
    report_examples = 0
    val_loss_hist = []
    num_trial = 0
    patience = 0

    model.train()

    for epoch in range(1, num_epochs + 1):
        cum_loss = 0
        cum_examples = 0

        for src, tgt in train_dl:
            # print(f'src: {src.size()}; tgt: {tgt.size()}')
            num_iter += 1

            batch_size = src.size(1)

            src = src.to(device)
            tgt = tgt.to(device)
            encoder_hidden = model.encoder.initHidden(
                batch_size=batch_size, device=device
            )

            # print(input_hidden.size())

            example_losses, _ = model(src, encoder_hidden, tgt)
            example_losses = -example_losses
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            bl = batch_loss.item()
            report_loss += bl
            report_examples += batch_size

            cum_loss += bl
            cum_examples += batch_size

            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            if num_iter % valid_niter == 0:
                val_loss = validate_model(model, val_dl, device)
                print(
                    f"Epoch {epoch}, iter {num_iter}, avg. train loss {report_loss/report_examples}, avg. val loss {val_loss}"
                )

                report_loss = 0
                report_examples = 0

                better_model = len(val_loss_hist) == 0 or val_loss < min(val_loss_hist)
                if better_model:
                    print(f"Saving model and optimizer to {model_save_path}")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        model_save_path,
                    )
                elif patience < max_num_patience:
                    patience += 1
                    print(f"hit patience {patience}")

                    if patience == max_num_patience:
                        num_trial += 1
                        print(f"hit #{num_trial} trial")
                        if num_trial == max_num_trial:
                            print("early stop!")
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]["lr"] * lr_decay
                        print(
                            f"load previously best model and decay learning rate to {lr}"
                        )

                        # load model
                        checkpoint = torch.load(model_save_path)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                        model = model.to(device)

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        # reset patience
                        patience = 0

                val_loss_hist.append(val_loss)

# %% ../nbs/02a_simple_correction_data.ipynb 23
class GreedySearchDecoder(torch.nn.Module):
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
        print(encoder_hidden.size())

        # Encoder part
        encoder_outputs = torch.zeros(
            batch_size, self.max_length, self.encoder.hidden_size, device=self.device
        )
        # print('encoder outputs size', encoder_outputs.size())

        for ei in range(input_length):
            # print(f'Index {ei}; input size: {input_tensor[ei].size()}; encoder hidden size: {encoder_hidden.size()}')
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden
            )
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

        decoder_input = torch.tensor(
            [[BOS_IDX] for _ in range(batch_size)], device=self.device
        )
        # print('decoder input size', decoder_input.size())

        all_tokens = torch.zeros(
            batch_size, self.max_length, device=self.device, dtype=torch.long
        )
        # print('all_tokens size', all_tokens.size())
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            # print('decoder input size:', decoder_input.size())
            # print('decoder input squeezed', decoder_input.clone().squeeze())

            # Record token
            all_tokens[:, di] = decoder_input.clone().squeeze(1)
            # print('all_tokens', all_tokens)

        return all_tokens

# %% ../nbs/02a_simple_correction_data.ipynb 25
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
