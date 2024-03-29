# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_icdar_data.ipynb.

# %% auto 0
__all__ = ['remove_label_and_nl', 'AlignedToken', 'tokenize_aligned', 'InputToken', 'leading_whitespace_offset',
           'get_input_tokens', 'Text', 'clean', 'normalized_ed', 'process_text', 'generate_data', 'extract_icdar_data',
           'get_intermediate_data', 'window', 'generate_sentences', 'process_input_ocr']

# %% ../nbs/00_icdar_data.ipynb 2
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from typing import Text as TypingText
from typing import Tuple
from zipfile import ZipFile

import edlib
import pandas as pd
from loguru import logger
from tqdm import tqdm

# %% ../nbs/00_icdar_data.ipynb 4
def remove_label_and_nl(line: str):
    return line.strip()[14:]

# %% ../nbs/00_icdar_data.ipynb 7
@dataclass
class AlignedToken:
    """Dataclass for storing aligned tokens"""

    ocr: str  # String in the OCR text
    gs: str  # String in the gold standard
    ocr_aligned: str  # String in the aligned OCR text (without aligmnent characters)
    gs_aligned: str  # String in the aligned GS text (without aligmnent characters)
    start: int  # The index of the first character in the OCR text
    len_ocr: int  # The lentgh of the OCR string

# %% ../nbs/00_icdar_data.ipynb 8
def tokenize_aligned(ocr_aligned: str, gs_aligned: str) -> List[AlignedToken]:
    """Get a list of AlignedTokens from the aligned OCR and GS strings"""

    ocr_cursor = 0
    start = 0

    ocr_token_chars: List[str] = []
    gs_token_chars: List[str] = []
    ocr_token_chars_aligned: List[str] = []
    gs_token_chars_aligned: List[str] = []

    tokens = []

    for ocr_aligned_char, gs_aligned_char in zip(ocr_aligned, gs_aligned):
        # print(ocr_aligned_char, gs_aligned_char, ocr_cursor)
        # The # character in ocr is not an aligment character!
        if ocr_aligned_char != "@":
            ocr_cursor += 1

        if ocr_aligned_char == " " and gs_aligned_char == " ":
            # print('TOKEN')
            # print('OCR:', repr(''.join(ocr_token_chars)))
            # print(' GS:', repr(''.join(gs_token_chars)))
            # print('start:', start_char)
            # ocr_cursor += 1

            # Ignore 'tokens' without representation in the ocr text
            # (these tokens do not consist of characters)
            ocr = ("".join(ocr_token_chars)).strip()
            if ocr != "":
                tokens.append(
                    AlignedToken(
                        ocr,
                        "".join(gs_token_chars),
                        "".join(ocr_token_chars_aligned),
                        "".join(gs_token_chars_aligned),
                        start,
                        len("".join(ocr_token_chars)),
                    )
                )
            start = ocr_cursor

            ocr_token_chars = []
            gs_token_chars = []
            ocr_token_chars_aligned = []
            gs_token_chars_aligned = []
        else:
            ocr_token_chars_aligned.append(ocr_aligned_char)
            gs_token_chars_aligned.append(gs_aligned_char)
            # The # character in ocr is not an aligment character!
            if ocr_aligned_char != "@":
                ocr_token_chars.append(ocr_aligned_char)
            if gs_aligned_char != "@" and gs_aligned_char != "#":
                gs_token_chars.append(gs_aligned_char)

    # Final token (if there is one)
    ocr = ("".join(ocr_token_chars)).strip()
    if ocr != "":
        tokens.append(
            AlignedToken(
                ocr,
                "".join(gs_token_chars),
                "".join(ocr_token_chars_aligned),
                "".join(gs_token_chars_aligned),
                start,
                len("".join(ocr_token_chars)),
            )
        )

    return tokens

# %% ../nbs/00_icdar_data.ipynb 11
@dataclass
class InputToken:
    """Dataclass for the tokenization within AlignedTokens"""

    ocr: str
    gs: str
    start: int
    len_ocr: int
    label: int

# %% ../nbs/00_icdar_data.ipynb 12
def leading_whitespace_offset(string: str) -> int:
    """
    Return the leading whitespace offset for an aligned ocr string

    If an aligned ocr string contains leading whitespace, the offset needs to be added
    to the start index of the respective input tokens.

    Args:
        string (str): aligned ocr string to determine the leading whitespace offset for

    Returns:
        int: leading whitespace offset for input tokens
    """
    offset = 0

    regex = r"^@*(?P<whitespace>\ +)"
    match = re.match(regex, string)
    if match:
        offset = len(match.group("whitespace"))

    return offset

# %% ../nbs/00_icdar_data.ipynb 17
def get_input_tokens(aligned_token: AlignedToken):
    """Tokenize an AlignedToken into subtokens and assign task 1 labels"""
    input_tokens = []
    if aligned_token.ocr == aligned_token.gs:
        input_tokens.append(
            InputToken(
                ocr=aligned_token.ocr,
                gs=aligned_token.gs,
                start=aligned_token.start,
                len_ocr=len(aligned_token.ocr),
                label=0,
            )
        )
    else:
        parts = aligned_token.ocr.split(" ")
        new_start = aligned_token.start
        offset = leading_whitespace_offset(aligned_token.ocr_aligned)
        for i, part in enumerate(parts):
            if i == 0:
                token = InputToken(
                    ocr=part,
                    gs=aligned_token.gs,
                    start=aligned_token.start,
                    len_ocr=len(part),
                    label=1,
                )
            else:
                token = InputToken(
                    ocr=part,
                    gs="",
                    start=new_start,
                    len_ocr=len(part),
                    label=2,
                )
                token.start += offset
            new_start += len(part) + 1
            input_tokens.append(token)
    return input_tokens

# %% ../nbs/00_icdar_data.ipynb 26
@dataclass
class Text:
    """Dataclass for storing a text in the ICDAR data format"""

    ocr_text: str
    tokens: list
    input_tokens: list
    score: float

# %% ../nbs/00_icdar_data.ipynb 27
def clean(string: str) -> str:
    """Remove alignment characters from a text"""
    string = string.replace("@", "")
    string = string.replace("#", "")

    return string

# %% ../nbs/00_icdar_data.ipynb 28
def normalized_ed(ed: int, ocr: str, gs: str) -> float:
    """Returns the normalized editdistance"""
    score = 0.0
    longest = max(len(ocr), len(gs))
    if longest > 0:
        score = ed / longest
    return score

# %% ../nbs/00_icdar_data.ipynb 29
def process_text(in_file: Path) -> Text:
    """Process a text from the ICDAR dataset

    Extract AlignedTokens, InputTokens, and calculate normalized editdistance.
    """
    with open(in_file) as f:
        lines = f.readlines()

    # The # character in ocr input is not an aligment character, but the @
    # character is!
    ocr_input = remove_label_and_nl(lines[0]).replace("@", "")
    ocr_aligned = remove_label_and_nl(lines[1])
    gs_aligned = remove_label_and_nl(lines[2])

    # print('ocr input:', ocr_input)
    # print('ocr aligned:', ocr_aligned)
    # print('gs aligned:',gs_aligned)

    tokens = tokenize_aligned(ocr_aligned, gs_aligned)

    # Check data
    for token in tokens:
        input_token = ocr_input[token.start : token.start + token.len_ocr]
        try:
            assert token.ocr == input_token.strip()
        except AssertionError:
            logger.warning(
                f"OCR != aligned OCR: Text: {str(in_file)}; ocr: {repr(token.ocr)}; "
                + f"ocr_input: {repr(input_token)}"
            )
            raise

    ocr = clean(ocr_aligned)
    gs = clean(gs_aligned)

    try:
        ed = edlib.align(gs, ocr)["editDistance"]
        score = normalized_ed(ed, ocr, gs)
    except UnicodeEncodeError:
        logger.warning(f"UnicodeEncodeError for text {in_file}; setting score to 1")
        score = 1

    input_tokens = []
    for token in tokens:
        for inp_tok in get_input_tokens(token):
            input_tokens.append(inp_tok)

    return Text(ocr_input, tokens, input_tokens, score)

# %% ../nbs/00_icdar_data.ipynb 35
def generate_data(in_dir: Path) -> Tuple[Dict[str, Text], pd.DataFrame]:
    """Process all texts in the dataset and return a dataframe with metadata"""

    data = {}

    file_languages = []
    file_subsets = []
    file_names = []
    scores = []
    num_tokens = []
    num_input_tokens = []

    in_files = []
    for language_dir in tqdm(in_dir.iterdir()):
        for text_file in language_dir.rglob("*.txt"):
            in_files.append(text_file)
    in_files.sort()

    for text_file in in_files:
        key = str(text_file.relative_to(in_dir))
        data[key] = process_text(text_file)

        language, subset, _ = key.split(os.sep)

        file_languages.append(language)
        file_subsets.append(subset)
        file_names.append(key)
        scores.append(data[key].score)
        num_tokens.append(len(data[key].tokens))
        num_input_tokens.append(len(data[key].input_tokens))
    md = pd.DataFrame(
        {
            "language": file_languages,
            "subset": file_subsets,
            "file_name": file_names,
            "score": scores,
            "num_tokens": num_tokens,
            "num_input_tokens": num_input_tokens,
        }
    )
    md.sort_values("file_name", inplace=True)
    md.reset_index(inplace=True, drop=True)
    return data, md

# %% ../nbs/00_icdar_data.ipynb 37
def extract_icdar_data(out_dir: TypingText, zip_file: TypingText) -> Tuple[Path, Path]:
    with ZipFile(zip_file, "r") as zip_object:
        zip_object.extractall(path=out_dir)

    # Copy Finnish data
    path = Path(out_dir)
    in_dir = path / "TOOLS_for_Finnish_data"
    inputs = {
        "evaluation": "ICDAR2019_POCR_competition_evaluation_4M_without_Finnish",
        "full": "ICDAR2019_POCR_competition_full_22M_without_Finnish",
        "training": "ICDAR2019_POCR_competition_training_18M_without_Finnish",
    }
    for from_dir, to_dir in inputs.items():
        for in_file in (in_dir / "output" / from_dir).iterdir():
            if in_file.is_file():
                out_file = path / to_dir / "FI" / "FI1" / in_file.name
                shutil.copy2(in_file, out_file)

    # Get paths for train and test data
    train_path = (
        Path(out_dir) / "ICDAR2019_POCR_competition_training_18M_without_Finnish"
    )
    test_path = (
        Path(out_dir) / "ICDAR2019_POCR_competition_evaluation_4M_without_Finnish"
    )

    return train_path, test_path


def get_intermediate_data(
    zip_file: TypingText,
) -> Tuple[Dict[str, Text], pd.DataFrame, Dict[str, Text], pd.DataFrame]:
    """Get the data and metadata for the train and test sets from the zip file."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path, test_path = extract_icdar_data(tmp_dir, zip_file)

        data, md = generate_data(train_path)
        data_test, md_test = generate_data(test_path)

    return (data, md, data_test, md_test)

# %% ../nbs/00_icdar_data.ipynb 39
def window(iterable, size=2):
    """Given an iterable, return all subsequences of a certain size"""
    i = iter(iterable)
    win = []
    for e in range(0, size):
        try:
            win.append(next(i))
        except StopIteration:
            break
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win

# %% ../nbs/00_icdar_data.ipynb 41
def _process_sequence(
    key: str,
    i: int,
    res,
    sents: List[List[str]],
    labels: List[List[int]],
    keys: List[str],
    start_tokens: List[int],
    scores: List[float],
    languages: List[str],
) -> Tuple[
    List[List[str]], List[List[int]], List[str], List[int], List[float], List[str]
]:
    ocr = [t.ocr for t in res]
    lbls = [t.label for t in res]
    gs = []
    for t in res:
        if t.gs != "":
            gs.append(t.gs)
    ocr_str = " ".join(ocr)
    gs_str = " ".join(gs)
    ed = edlib.align(ocr_str, gs_str)["editDistance"]
    score = normalized_ed(ed, ocr_str, gs_str)

    if len(ocr_str) > 0:
        sents.append(ocr)
        labels.append(lbls)
        keys.append(key)
        start_tokens.append(i)
        scores.append(score)
        languages.append(key[:2])
    else:
        logger.info(f'Empty sample for text "{key}"')
        logger.info(f"ocr_str: {ocr_str}")
        logger.info(f"start token: {i}")

    return (sents, labels, keys, start_tokens, scores, languages)


def generate_sentences(
    df: pd.DataFrame, data: Dict[str, Text], size: int = 15, step: int = 10
) -> pd.DataFrame:
    """Generate sequences of a certain length and possible overlap"""
    sents: List[List[str]] = []
    labels: List[List[int]] = []
    keys: List[str] = []
    start_tokens: List[int] = []
    scores: List[float] = []
    languages: List[str] = []

    for _, row in tqdm(df.iterrows()):
        key = row.file_name
        tokens = data[key].input_tokens

        # print(len(tokens))
        # print(key)
        for i, res in enumerate(window(tokens, size=size)):
            if i % step == 0:
                (
                    sents,
                    labels,
                    keys,
                    start_tokens,
                    scores,
                    languages,
                ) = _process_sequence(
                    key, i, res, sents, labels, keys, start_tokens, scores, languages
                )
        # Add final sequence
        (sents, labels, keys, start_tokens, scores, languages) = _process_sequence(
            key, i, res, sents, labels, keys, start_tokens, scores, languages
        )

    output = pd.DataFrame(
        {
            "key": keys,
            "start_token_id": start_tokens,
            "score": scores,
            "tokens": sents,
            "tags": labels,
            "language": languages,
        }
    )

    # Adding the final sequence may lead to duplicate rows. Remove those
    output.drop_duplicates(
        subset=["key", "start_token_id"], keep="first", inplace=True, ignore_index=True
    )

    return output

# %% ../nbs/00_icdar_data.ipynb 43
def process_input_ocr(text: str) -> Text:
    """Generate Text object for OCR input text (without aligned gold standard)"""
    tokens = []
    for match in re.finditer(r"\b\S+(\s|$)", text):
        ocr = match.group().strip()
        gs = ocr
        start = match.start()
        len_ocr = len(ocr)
        label = 0

        tokens.append(
            InputToken(
                ocr=ocr,
                gs=gs,
                start=start,
                len_ocr=len_ocr,
                label=label,
            )
        )
    return Text(text, tokens=[], input_tokens=tokens, score=-1)
