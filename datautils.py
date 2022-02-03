import os
from dataclasses import dataclass
import edlib
import pandas as pd
from tqdm import tqdm

from loguru import logger


def remove_label_and_nl(line):
    return line.strip()[14:]


def normalized_ed(ed, ocr, gs):
    score = 0.0
    l = max(len(ocr), len(gs))
    if l > 0:
        score = ed / l
    return score


@dataclass
class AlignedToken:
    ocr: str
    gs: str
    ocr_aligned: str
    gs_aligned: str
    start: int
    len_ocr: int


@dataclass
class InputToken:
    ocr: str
    gs: str
    start: int
    label: int


def get_input_tokens(aligned_token):
    if aligned_token.ocr == aligned_token.gs:
            yield InputToken(aligned_token.ocr, aligned_token.gs, aligned_token.start, 0)
    else:
        parts = aligned_token.ocr.split(' ')
        new_start = aligned_token.start
        for i, part in enumerate(parts):
            if i == 0:
                yield InputToken(part, aligned_token.gs, aligned_token.start, 1)
            else:
                yield InputToken(part, '', new_start, 2)
            new_start += len(part) + 1


def tokenize_aligned(ocr_aligned, gs_aligned):

    ocr_cursor = 0
    start = 0

    ocr_token_chars = []
    gs_token_chars = []
    ocr_token_chars_aligned = []
    gs_token_chars_aligned = []

    tokens = []

    for ocr_aligned_char, gs_aligned_char in zip(ocr_aligned, gs_aligned):
        #print(ocr_aligned_char, gs_aligned_char, ocr_cursor)
        if ocr_aligned_char != '@' and ocr_aligned_char != '#':
            ocr_cursor += 1

        if ocr_aligned_char == ' ' and gs_aligned_char == ' ':
            #print('TOKEN')
            #print('OCR:', repr(''.join(ocr_token_chars)))
            #print(' GS:', repr(''.join(gs_token_chars)))
            #print('start:', start_char)
            #ocr_cursor += 1

            # Ignore 'tokens' without representation in the ocr text 
            # (these tokens do not consist of characters) 
            ocr = (''.join(ocr_token_chars)).strip()
            if ocr != '':
                tokens.append(AlignedToken(ocr, 
                                          ''.join(gs_token_chars), 
                                          ''.join(ocr_token_chars_aligned), 
                                          ''.join(gs_token_chars_aligned),
                                          start,
                                          len(''.join(ocr_token_chars))))
            start = ocr_cursor

            ocr_token_chars = []
            gs_token_chars = []
            ocr_token_chars_aligned = []
            gs_token_chars_aligned = []
        else:
            ocr_token_chars_aligned.append(ocr_aligned_char)
            gs_token_chars_aligned.append(gs_aligned_char)
            if ocr_aligned_char != '@' and ocr_aligned_char != '#':
                ocr_token_chars.append(ocr_aligned_char)
            if gs_aligned_char != '@' and gs_aligned_char != '#':
                gs_token_chars.append(gs_aligned_char)
    
    ocr = (''.join(ocr_token_chars)).strip()
    if ocr != '':
        tokens.append(AlignedToken(ocr, 
                                   ''.join(gs_token_chars), 
                                   ''.join(ocr_token_chars_aligned), 
                                   ''.join(gs_token_chars_aligned),
                                   start,
                                   len(''.join(ocr_token_chars))))

    return tokens


def window(iterable, size=2):
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


@dataclass
class Text:
    tokens: list
    input_tokens: list
    score: float


def clean(string):
    string = string.replace('@', '')
    string = string.replace('#', '')

    return string


def process_text(in_file):
    with open(in_file) as f:
        lines = f.readlines()

    ocr_input = clean(remove_label_and_nl(lines[0]))
    ocr_aligned = remove_label_and_nl(lines[1])
    gs_aligned = remove_label_and_nl(lines[2])

    #print('ocr input:', ocr_input)
    #print('ocr aligned:', ocr_aligned)
    #print('gs aligned:',gs_aligned)

    tokens = tokenize_aligned(ocr_aligned, gs_aligned)

    # Check data
    for token in tokens:
        input_token = ocr_input[token.start:token.start+token.len_ocr]
        try:
            assert token.ocr == input_token.strip()
        except AssertionError:
            print(f'Text: {str(in_file)}; ocr: {repr(token.ocr)}; ocr_input: {repr(input_token)}')
            raise

    ocr = clean(ocr_aligned)
    gs = clean(gs_aligned)

    try:
        ed = edlib.align(gs, ocr)['editDistance']
        score = normalized_ed(ed, ocr, gs)
    except UnicodeEncodeError:
        logger.warning(f'UnicodeEncodeError for text {in_file}; setting score to 1')
        score = 1

    input_tokens = []
    for token in tokens:
        for inp_tok in get_input_tokens(token):
            input_tokens.append(inp_tok)
    
    return Text(tokens, input_tokens, score)


def generate_data(in_dir):

    data = {}

    file_languages = []
    file_names = []
    scores = []
    num_tokens = []
    num_input_tokens = []

    for language_dir in tqdm(in_dir.iterdir()):
        #print(language_dir.stem)
        language = language_dir.stem
        
        for text_file in language_dir.rglob('*.txt'):
            #print(text_file)
            #print(text_file.relative_to(in_dir))
            key = str(text_file.relative_to(in_dir))
            data[key] = process_text(text_file)

            file_languages.append(language)
            file_names.append(key)
            scores.append(data[key].score)
            num_tokens.append(len(data[key].tokens))
            num_input_tokens.append(len(data[key].input_tokens))
    md = pd.DataFrame({'language': file_languages, 
                    'file_name': file_names,
                    'score': scores,
                    'num_tokens': num_tokens,
                    'num_input_tokens': num_input_tokens})
    return data, md


def generate_sentences(df, data, size=15, step=10):
    sents = []
    labels = []
    keys = []
    start_tokens = []
    scores = []
    languages = []

    for idx, row in tqdm(df.iterrows()):
        key = row.file_name
        tokens = data[key].input_tokens

        # print(len(tokens))
        # print(key)
        for i, res in enumerate(window(tokens, size=size)):
            if i % step == 0:
                ocr = [t.ocr for t in res]
                lbls = [t.label for t in res]
                gs = []
                for t in res:
                    if t.gs != '':
                        gs.append(t.gs)
                ocr_str = ' '.join(ocr)
                gs_str = ' '.join(gs)
                ed = edlib.align(ocr_str, gs_str)['editDistance']
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
                    logger.info(f'ocr_str: {ocr_str}')
                    logger.info(f'start token: {i}')
    data = pd.DataFrame({
        'key': keys,
        'start_token_id': start_tokens,
        'score': scores,
        'tokens': sents,
        'tags': labels,
        'language': languages
    })

    return data
