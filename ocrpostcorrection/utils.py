# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_utils.ipynb.

# %% auto 0
__all__ = ['maxNbCandidate', 'set_seed', 'predictions_to_labels', 'separate_subtoken_predictions', 'merge_subtoken_predictions',
           'gather_token_predictions', 'labels2label_str', 'extract_icdar_output', 'predictions2icdar_output',
           'create_entity', 'extract_entity_output', 'predictions2entity_output', 'create_perfect_icdar_output',
           'EvalContext', 'reshape_input_errors', 'runEvaluation', 'read_results',
           'icdar_output2simple_correction_dataset_df', 'aggregate_results', 'aggregate_ed_results', 'reduce_dataset']

# %% ../nbs/03_utils.ipynb 2
import codecs
import collections
import itertools
import json
import os
import random
import re
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from transformers import AutoTokenizer
import torch

from .icdar_data import Text

# %% ../nbs/03_utils.ipynb 5
def set_seed(seed: int) -> None:
    """Set the random seed in Python std library and pytorch

    Args:
        seed (int): Value of the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)  # For use with torch.device("mps") (on Mac)

# %% ../nbs/03_utils.ipynb 8
def predictions_to_labels(predictions):
    return np.argmax(predictions, axis=2)

# %% ../nbs/03_utils.ipynb 11
def separate_subtoken_predictions(word_ids, preds):
    # print(len(word_ids), word_ids)
    result = defaultdict(list)
    for word_idx, p_label in zip(word_ids, preds):
        # print(word_idx, p_label)
        if word_idx is not None:
            result[word_idx].append(p_label)
    return dict(result)

# %% ../nbs/03_utils.ipynb 13
def merge_subtoken_predictions(subtoken_predictions):
    token_level_predictions = []
    for word_idx, preds in subtoken_predictions.items():
        token_label = 0
        c = Counter(preds)
        # print(c)
        if c[1] > 0 and c[1] >= c[2]:
            token_label = 1
        elif c[2] > 0 and c[2] >= c[1]:
            token_label = 2

        token_level_predictions.append(token_label)
    return token_level_predictions

# %% ../nbs/03_utils.ipynb 15
def gather_token_predictions(preds):
    """Gather potentially overlapping token predictions"""
    labels = defaultdict(list)

    # print(len(text.input_tokens))
    # print(preds)
    for start, lbls in preds.items():
        for i, label in enumerate(lbls):
            labels[int(start) + i].append(label)
    # print('LABELS')
    # print(labels)
    return dict(labels)

# %% ../nbs/03_utils.ipynb 17
def labels2label_str(labels, text_key):
    label_str = []
    i = 0

    for token in labels:
        # print(i, token)
        while i < token:
            logger.warning(f'Missing predictions for token {i} in "{text_key}"')
            # Predictions are missing (input text was truncated)
            # Add 0 to make sure token indices remain correct
            label_str.append("0")
            i += 1

        if 2 in labels[i]:
            label_str.append("2")
        elif 1 in labels[i]:
            label_str.append("1")
        else:
            label_str.append("0")
        i += 1

    label_str = "".join(label_str)
    return label_str

# %% ../nbs/03_utils.ipynb 21
def extract_icdar_output(label_str, input_tokens):
    keys = {}
    started = False
    start_idx = -1
    num_tokens = 0
    for input_token, label in zip(input_tokens, label_str):
        if label == "1":
            if started:
                keys[start_idx] = num_tokens
                started = False
                start_idx = -1
                num_tokens = 0

            started = True
            start_idx = input_token.start
            num_tokens += 1
        elif label == "2":
            if not started:
                started = True
                start_idx = input_token.start
            num_tokens += 1
        else:
            # label = '0'
            if started:
                keys[start_idx] = num_tokens
                started = False
                start_idx = -1
                num_tokens = 0
    # Add final ocr mistake
    if started:
        keys[start_idx] = num_tokens

    text_output = {}
    for offset, num_tokens in keys.items():
        text_output[f"{offset}:{num_tokens}"] = {}

    return text_output

# %% ../nbs/03_utils.ipynb 29
def _predictions2label_str(samples, predictions, tokenizer):
    """Convert predictions into label strings"""
    # print('samples', len(samples))
    # print(samples)
    # print(samples[0].keys())
    # for sample in samples:
    #    print(sample.keys())

    tokenized_samples = tokenizer(
        samples["tokens"], truncation=True, is_split_into_words=True
    )
    # print(samples)

    # for sample in samples:
    #    print(sample.keys())

    # convert predictions to labels (label_ids)
    # p = np.argmax(predictions, axis=2)
    # print(p)

    converted = defaultdict(dict)

    for i, (sample, preds) in enumerate(zip(samples, predictions)):
        # print(sample.keys())
        # label = sample['tags']
        # print(label)
        # print(len(preds), preds)
        word_ids = tokenized_samples.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        result = separate_subtoken_predictions(word_ids, preds)
        new_tags = merge_subtoken_predictions(result)

        # print('pred', len(new_tags), new_tags)
        # print('tags', len(label), label)

        # print(sample)
        # print(sample['key'], sample['start_token_id'])
        converted[sample["key"]][sample["start_token_id"]] = new_tags

    output = {}
    for key, preds in converted.items():
        labels = defaultdict(list)
        # print(key)
        labels = gather_token_predictions(preds)
        label_str = labels2label_str(labels, key)
        output[key] = label_str

    return output

# %% ../nbs/03_utils.ipynb 30
def predictions2icdar_output(samples, predictions, tokenizer, data_test):
    """Convert predictions into icdar output format"""
    converted = _predictions2label_str(samples, predictions, tokenizer)
    output = {}
    for key, label_str in converted.items():
        try:
            text = data_test[key]
            output[key] = extract_icdar_output(label_str, text.input_tokens)
        except KeyError:
            logger.warning(f"No data found for text {key}")

    return output

# %% ../nbs/03_utils.ipynb 35
def create_entity(entity_tokens):
    start = entity_tokens[0].start
    end = entity_tokens[-1].start + entity_tokens[-1].len_ocr
    word = " ".join([token.ocr for token in entity_tokens])
    return {"entity": "OCR mistake", "word": word, "start": start, "end": end}

# %% ../nbs/03_utils.ipynb 40
def extract_entity_output(label_str: str, input_tokens):
    """Convert label string to the entity output format"""
    entity_tokens = []
    entities = []
    for token, label in zip(input_tokens, label_str):
        if label == "0":
            if len(entity_tokens) > 0:
                entities.append(create_entity(entity_tokens))
                entity_tokens = []
        elif label == "1":
            if len(entity_tokens) > 0:
                entities.append(create_entity(entity_tokens))
                entity_tokens = []
            entity_tokens.append(token)
        elif label == "2":
            entity_tokens.append(token)

    # Final token
    if len(entity_tokens) > 0:
        entities.append(create_entity(entity_tokens))

    return entities

# %% ../nbs/03_utils.ipynb 42
def predictions2entity_output(samples, predictions, tokenizer, data_test):
    """Convert predictions into entity output format"""
    converted = _predictions2label_str(samples, predictions, tokenizer)
    output = {}
    for key, label_str in converted.items():
        try:
            text = data_test[key]
            output[key] = extract_entity_output(label_str, text.input_tokens)
        except KeyError:
            logger.warning(f"No data found for text {key}")

    return output

# %% ../nbs/03_utils.ipynb 44
def create_perfect_icdar_output(data):
    output = {}
    for key, text_obj in data.items():
        label_str = "".join([str(t.label) for t in text_obj.input_tokens])
        output[key] = extract_icdar_output(label_str, data[key].input_tokens)
    return output

# %% ../nbs/03_utils.ipynb 47
maxNbCandidate = 6

# %% ../nbs/03_utils.ipynb 48
################# CLASS FOR STORING CURRENT FILE CONTEXT  ################
class EvalContext:
    # Default symbols used for the alignment and for ignoring some tokens
    charExtend = r"@"
    charIgnore = r"#"

    # Different texts versions provided
    ocrAligned, gsAligned, ocrOriginal = "", "", ""

    # Alignment map (ocrOriginal <=> ocrAligned and gsAligned)
    aMap = []

    def __init__(self, filePath, verbose=False):
        assert os.path.exists(filePath), "[ERROR] : File %s not found !" % filePath

        self.filePath = filePath
        self.verbose = verbose

        # Load file data
        with open(filePath, "r") as f:
            text = f.read().strip()
            self.ocrOriginal, self.ocrAligned, self.gsAligned = [
                txt[14:] for txt in re.split(r"\r?\n", text)
            ]
            # text.strip() removes trailing space from gs aligned, but not from the other texts.
            # This causes problems when calculating recall. The solution is to also remove
            # trailing space from ocr original and ocr aligned.
            self.ocrOriginal = self.ocrOriginal.rstrip()
            self.ocrAligned = self.ocrAligned.rstrip()

            if self.charExtend in self.ocrOriginal:
                print(f"{self.charExtend} found in ocrOriginal. Removing...")
                self.ocrOriginal = self.ocrOriginal.replace(self.charExtend, "")

        # Check file integrity
        assert (
            self.ocrOriginal == re.sub(self.charExtend, "", self.ocrAligned).rstrip()
        ), (
            '[ERROR] : [OCR_aligned] without "%s" doesn\'t correspond to [OCR_toInput] in file %s'
            % (self.charExtend, filePath)
        )

        # Build the alignment map
        self.aMap = [
            x.start() - i
            for i, x in enumerate(re.finditer(self.charExtend + r"|$", self.ocrAligned))
        ]

        # print("%s\n%s\n%s" % (self.ocrOriginal, self.ocrAligned, self.gsAligned))

    # Get the alignment shift for a position in the orginal OCR to the corresponding postion in the aligned OCR/GS
    def get_aligned_shift(self, posOriginal):
        return next((i for i, e in enumerate(self.aMap) if e >= posOriginal), 0)

    # Get the alignment shift for a position in the orginal OCR to the corresponding postion in the aligned OCR/GS
    def get_original_shift(self, posAligned):
        return self.ocrAligned.count(self.charExtend, 0, posAligned)

    # Get bounds in "Aligned OCR/GS" from a token position in the non-aligned OCR.
    def get_aligned_token_bounds(self, tokenPos, nbToken=1):
        assert (tokenPos == 0) or (
            self.ocrOriginal[tokenPos - 1] == " "
        ), "[ERROR] : %d is not a token start position (%s)" % (tokenPos, self.filePath)

        alignedPos = tokenPos + self.get_aligned_shift(tokenPos)
        seqLen = nbToken - 1  # Init with number of spaces
        iterOcrAlignedSpace = re.finditer(r"$|\ ", self.ocrAligned[alignedPos:])
        for nbt in range(nbToken):
            matchSpace = next(iterOcrAlignedSpace, None)
            if matchSpace is None:
                print(
                    "[WARNING] : At pos %d, could not iterate forward over tokens, end of the sequence reached"
                    % tokenPos
                )
                break
            seqLen = matchSpace.start()  # Look for last space before next token

        return alignedPos, seqLen

    # Get statistics (erroneous tokens' posisitions, corrections, ect...) on errors
    def get_errors_stats(self):
        # results = {}
        nbTokens, nbErrTokens, nbErrTokensAlpha = 0, 0, 0

        # Iterate over GS tokens
        lastTokenPos = 0
        for spacePos in re.finditer(r"$|\ ", self.gsAligned):
            tokenEndPos = spacePos.start()
            tokenInOcr = re.sub(
                self.charExtend, "", self.ocrAligned[lastTokenPos:tokenEndPos+1]
            )
            tokenInGs = re.sub(
                self.charExtend, "", self.gsAligned[lastTokenPos:tokenEndPos+1]
            )

            if self.verbose:
                print(f"Indices aligned: start: {lastTokenPos}, end: {tokenEndPos}")
                print(f"Token in ocr: '{tokenInOcr}' (aligned: '{self.ocrAligned[lastTokenPos:tokenEndPos+1]}')")
                print(f"Token in gs: '{tokenInGs}' (aligned: '{self.gsAligned[lastTokenPos:tokenEndPos+1]}')")
                print('---')

            if self.charIgnore in tokenInGs:
                lastTokenPos = tokenEndPos + 1
                continue

            # if (tokenInOcr != tokenInGs):
            #    results[lastTokenPos] = [tokenInOcr, tokenInGs]
            #    #print("_%s_%s_" % (tokenInOcr, tokenInGs))

            lastTokenPos = tokenEndPos + 1

            nbTokens += 1
            nbErrTokens += tokenInOcr != tokenInGs
            nbErrTokensAlpha += (
                tokenInOcr != tokenInGs
            ) and tokenInGs.strip().isalpha()

        return nbTokens, nbErrTokens, nbErrTokensAlpha

    def collectErrorPos(self):
        errorList = []

        # Add tolerance to hyphens : work on a new GS where hyphens founds in OCR are considered to be ignored in task1
        gsAlignedHyphensIgnored = self.gsAligned
        for hToken in re.finditer(
            r"[^ ]*((\ ?-[^ ])|([^ ]-\ ?))[^ ]*", self.ocrAligned
        ):
            gsAlignedHyphensIgnored = (
                gsAlignedHyphensIgnored[: hToken.start()]
                + self.charIgnore * (hToken.end() - hToken.start())
                + gsAlignedHyphensIgnored[hToken.end() :]
            )

        gsSpacePos = set(
            [
                spacePos.start()
                for spacePos in re.finditer(r"^|$|\ ", gsAlignedHyphensIgnored)
            ]
        )
        ocrSpacePos = set(
            [spacePos.start() for spacePos in re.finditer(r"^|$|\ ", self.ocrAligned)]
        )
        commonSpacePos = sorted(gsSpacePos.intersection(ocrSpacePos))
        # print(commonSpacePos)

        for i in range(len(commonSpacePos) - 1):
            tokenStartPos = commonSpacePos[i] + 1 * (
                gsAlignedHyphensIgnored[commonSpacePos[i]] == " "
            )
            tokenEndPos = commonSpacePos[i + 1]

            tokenInGs = re.sub(
                self.charExtend, "", gsAlignedHyphensIgnored[tokenStartPos:tokenEndPos]
            )
            tokenInOcr = re.sub(
                self.charExtend, "", self.ocrAligned[tokenStartPos:tokenEndPos]
            )

            # Get not aligned pos
            tokenStartPosOriginal = tokenStartPos - self.get_original_shift(
                tokenStartPos
            )

            # Ignore the "#" in GS
            if not (self.charIgnore in tokenInGs):
                if tokenInOcr != tokenInGs:
                    # print("%d:%d|%d=%s=>%s" % (tokenStartPosOriginal, tokenInOcr.count(" ")+1, tokenStartPos , tokenInOcr, tokenInGs))
                    errorList.append(
                        "%d:%d" % (tokenStartPosOriginal, tokenInOcr.count(" ") + 1)
                    )
                    # errorList.append("%d:%d|%d=%s=>%s" % (tokenStartPosOriginal, tokenInOcr.count(" ") + 1, tokenStartPos, tokenInOcr, tokenInGs))

        return errorList

    def task1_eval(self, inputErroneousTokens, print_sets=False):
        # Add tolerance to hyphens : work on a new GS where hyphens founds in OCR are considered to be ignored in task1
        gsAlignedHyphensIgnored = self.gsAligned
        for hToken in re.finditer(
            r"[^ ]*((\ ?-[^ ])|([^ ]-\ ?))[^ ]*", self.ocrAligned
        ):
            gsAlignedHyphensIgnored = (
                gsAlignedHyphensIgnored[: hToken.start()]
                + self.charIgnore * (hToken.end() - hToken.start())
                + gsAlignedHyphensIgnored[hToken.end() :]
            )

        # 1) Prepare input results : unfold overlapping n tokens in tokenPosErr
        detectedErrPosUnfolded = {}
        for errPos, val in inputErroneousTokens.items():
            tokenStartPos = 0
            iterTokens = re.finditer(r"$|\ ", self.ocrOriginal[errPos:])
            for t in range(val["nbToken"]):
                alignedPos, seqLen = self.get_aligned_token_bounds(
                    (errPos + tokenStartPos), 1
                )
                rawTokenInAlignedGs = gsAlignedHyphensIgnored[
                    alignedPos : (alignedPos + seqLen + 1)
                ]

                if not (self.charIgnore in rawTokenInAlignedGs):
                    # Check if there is no overlapping errors/corrections
                    assert alignedPos not in detectedErrPosUnfolded, (
                        "[ERROR] : Error at pos %d is overlapping another given error ! Pay attention to the number of overlapping tokens."
                        % (errPos + tokenStartPos)
                    )
                    detectedErrPosUnfolded[alignedPos] = [
                        rawTokenInAlignedGs,
                        val["candidates"],
                    ]

                tokenEndMatch = next(iterTokens, None)
                if tokenEndMatch is None:
                    break

                tokenStartPos = tokenEndMatch.start() + 1

        if self.verbose:
            EvalContext.printDicoSortedByKey(
                detectedErrPosUnfolded, "1) detectedErrPosUnfolded"
            )

        # 2) Prepare real results
        realErrPosUnfolded = {}
        iterTokens = re.finditer(r"$|\ ", self.ocrAligned)
        tokenStartPos = 0
        tokenEndMatch = next(iterTokens, None)

        while tokenEndMatch is not None:
            tokenEndPos = tokenEndMatch.start() + 1  # Include following char

            tokenInGs = re.sub(
                self.charExtend,
                "",
                gsAlignedHyphensIgnored[max(0, tokenStartPos - 1) : tokenEndPos],
            )
            tokenInOcr = re.sub(
                self.charExtend,
                "",
                self.ocrAligned[max(0, tokenStartPos - 1) : tokenEndPos],
            )

            # Ignore the "#" in GS
            if not (self.charIgnore in tokenInGs):
                if tokenInOcr != tokenInGs:
                    realErrPosUnfolded[tokenStartPos] = [tokenInOcr, tokenInGs]

            tokenStartPos = tokenEndPos
            tokenEndMatch = next(iterTokens, None)

        if self.verbose:
            EvalContext.printDicoSortedByKey(
                realErrPosUnfolded, "2) realErrPosUnfolded"
            )

        realErrPos = set(realErrPosUnfolded.keys())

        # 3) Compute classical metrics
        setErrPos = set(detectedErrPosUnfolded.keys())
        errTP = len(setErrPos.intersection(realErrPos))  # TruePositive
        errFP = len(setErrPos.difference(realErrPos))  # TrueNegative
        errFN = len(realErrPos.difference(setErrPos))  # FalseNegative

        # Possible division per 0
        prec = (errTP / float(errTP + errFP)) if (errTP + errFP) > 0 else 0
        recall = (errTP / float(errTP + errFN)) if (errTP + errFN) > 0 else 0
        fmes = (
            (2.0 * float(prec * recall) / float(prec + recall))
            if (prec + recall) > 0
            else 0
        )

        # Debug test
        if self.verbose or print_sets:
            print(
                "TASK 1) ErrTP %d / errFP %d / errFN %d /" % (errTP, errFP, errFN)
                + " prec %0.2f / recall %0.2f / fmes %0.2f" % (prec, recall, fmes)
            )
        if print_sets:
            print("False positives:", setErrPos.difference(realErrPos))
            print("False negatives:", realErrPos.difference(setErrPos))

        return prec, recall, fmes

    def task2_eval(self, inputErroneousTokens, useFirstCandidateOnly=False):
        # Init list of tokens' levenshtein distances
        originalDistance, correctedDistance = [0], [0]
        nbSymbolsConsidered = 0

        # Get tokens (or sequences including hyphens)
        splitRegExp = r"(?=([^-]\ [^-]))"  # Now supporting overlapping
        spaceOCR = set(
            [sp.start() + 1 for sp in re.finditer(splitRegExp, self.ocrAligned)]
        )
        spaceGS = set(
            [sp.start() + 1 for sp in re.finditer(splitRegExp, self.gsAligned)]
        )

        # Collect running tokens defined in correction...
        spaceCorToRemove = []
        inputErroneousTokensAligned = {}
        for p, details in inputErroneousTokens.items():
            posAligned = p + self.get_aligned_shift(p)
            inputErroneousTokensAligned[posAligned] = details
            for m in itertools.islice(
                re.finditer(r"$|\ ", self.ocrAligned[posAligned:]),
                details["nbToken"] - 1,
            ):
                spaceCorToRemove.append(m.start() + posAligned)

        # print("spaceCommon %s" % str(sorted(spaceOCR.intersection(spaceGS))))
        # print("spaceCorToRemove %s" % str(sorted(spaceCorToRemove)))

        spaceCommon = (spaceOCR.intersection(spaceGS)).difference(spaceCorToRemove)
        spaceCommon.add(len(self.ocrAligned))

        # Iterate over comparable sequences
        lastTokenStartPos = 0
        for tokenEndPos in sorted(spaceCommon):
            tokenInGs = self.gsAligned[lastTokenStartPos:tokenEndPos]
            tokenInOcr = self.ocrAligned[lastTokenStartPos:tokenEndPos]

            # Get corrections concerned by this sequence :
            tokensPosToCorrect = set(
                range(lastTokenStartPos, tokenEndPos)
            ).intersection(inputErroneousTokensAligned.keys())

            listCombinaisons = []
            for p in tokensPosToCorrect:
                listCombinaisons.append(
                    [
                        [p, c]
                        for c, w in inputErroneousTokensAligned[p]["candidates"].items()
                    ]
                )

            tokenProposed = {}
            for combi in itertools.product(*iter(listCombinaisons)):
                # Default
                corToken = list(tokenInOcr)

                prodWeight = 1.0
                for pc in sorted(combi, key=lambda k: k[0], reverse=True):
                    offsetStart = (
                        inputErroneousTokensAligned[pc[0]]["boundsAligned"][0]
                        - lastTokenStartPos
                    )
                    offsetEnd = (
                        offsetStart
                        + inputErroneousTokensAligned[pc[0]]["boundsAligned"][1]
                    )

                    corToken[offsetStart:offsetEnd] = pc[1]
                    prodWeight = (
                        prodWeight
                        * inputErroneousTokensAligned[pc[0]]["candidates"][pc[1]]
                    )

                tokenProposed[
                    re.sub(self.charExtend, "", "".join(corToken))
                ] = prodWeight

            # Fix GC 15/06/2017
            if len(tokenProposed) == 0:
                tokenProposed = {tokenInOcr: 1.0}

            # In case we want to consider only the highest candidate
            if useFirstCandidateOnly and len(tokenProposed) > 0:
                tokenProposed = {max(tokenProposed, key=tokenProposed.get): 1.0}

            # Consider results only if no "#" found in the token (or tokens sequence).
            if not (self.charIgnore in tokenInGs):
                # Update damerau_levenshtein_distance distances' lists
                ignoreList = [" -", "- ", "-", self.charExtend]
                originalDistance.append(
                    EvalContext.damerau_levenshtein(
                        tokenInOcr, tokenInGs, ignoreList=ignoreList
                    )
                )

                weightedSum = sum(
                    [
                        EvalContext.damerau_levenshtein(
                            token, tokenInGs, ignoreList=ignoreList
                        )
                        * float(w)
                        for token, w in tokenProposed.items()
                    ]
                )
                correctedDistance.append(weightedSum)
                nbSymbolsConsidered += len(tokenInOcr)

            else:
                # print("[IGNORED] Token _%s_%s_ => %s" % (tokenInOcr, tokenInGs, tokenProposed))
                pass

            # print("(%d=>%d) %s | %s | %s " % (lastTokenStartPos,tokenEndPos, re.sub(self.charExtend, "", tokenInOcr), re.sub(self.charExtend, "", tokenInGs), str(tokenProposed) ))
            lastTokenStartPos = tokenEndPos + 1

        if self.verbose:
            print(
                "TASK 2) correctedDistance %d vs originalDistance %d"
                % (sum(correctedDistance), sum(originalDistance))
            )

        # print(correctedDistance)

        return sum(correctedDistance), sum(originalDistance), nbSymbolsConsidered

    # --- Damerau-Levenshtein distance between 2 strings ---
    # Slightly modified version of https://github.com/jamesturk/jellyfish
    # under Copyright 2015, James TurkJames Turk, Sunlight Foundation
    # with LICENSE BSD 2: https://github.com/jamesturk/jellyfish/blob/master/LICENSE
    @staticmethod
    def damerau_levenshtein(s1, s2, ignoreList=[" -", "- ", "-"]):
        # Add tolerence on some characters (e.g. hyphens) cause GS is not always perfect.
        s1 = re.sub("(" + ")|(".join(ignoreList) + ")", "", s1)
        s2 = re.sub("(" + ")|(".join(ignoreList) + ")", "", s2)

        if s1 == s2:
            return 0

        len1 = len(s1)
        len2 = len(s2)
        infinite = len1 + len2

        # character array
        da = collections.defaultdict(int)

        # distance matrix
        score = [[0] * (len2 + 2) for x in range(len1 + 2)]

        score[0][0] = infinite
        for i in range(0, len1 + 1):
            score[i + 1][0] = infinite
            score[i + 1][1] = i
        for i in range(0, len2 + 1):
            score[0][i + 1] = infinite
            score[1][i + 1] = i

        for i in range(1, len1 + 1):
            db = 0
            for j in range(1, len2 + 1):
                i1 = da[s2[j - 1]]
                j1 = db
                cost = 1
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                    db = j

                score[i + 1][j + 1] = min(
                    score[i][j] + cost,
                    score[i + 1][j] + 1,
                    score[i][j + 1] + 1,
                    score[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
                )
            da[s1[i - 1]] = i

        return score[len1 + 1][len2 + 1]

    # For debugging
    @staticmethod
    def printDicoSortedByKey(d, dicoName="Dico"):
        sortedKeysDic = list(d.keys())
        sortedKeysDic.sort()
        print("\n#########---- Print Sorted %s ----######### " % dicoName)
        for k in sortedKeysDic:
            print("%s:%s" % (str(k), str(d[k])))

# %% ../nbs/03_utils.ipynb 51
def reshape_input_errors(tokenPosErr, evalContext, verbose=False):
    # Store tokens' positions in mem
    tokensPos = [0] + [
        spacePos.start() + 1 for spacePos in re.finditer(r"\ ", evalContext.ocrOriginal)
    ]

    # 1) Check JSON result format (ex: positions correspond to tokens)
    # 2) Reshape data "pos":{"nbTokens":..., "boundsAligned":..., candidates:... }"
    # 3) Locally normalize candidates' weights id needed
    tokenPosErrReshaped = {}
    for pos_nbtokens, candidates in tokenPosErr.items():
        pos, nbOverlappingToken = [int(i) for i in pos_nbtokens.split(":")]
        boundsAligned = evalContext.get_aligned_token_bounds(pos, nbOverlappingToken)

        # Check pos targets a existing token (first char)
        assert pos in tokensPos, (
            "[ERROR] : Error at pos %s does not target the first char of a token (space separated sequences)."
            % pos
        )

        assert evalContext.ocrOriginal[pos:].count(" ") >= nbOverlappingToken - 1, (
            "[ERROR] : Error at pos %d spreads overs %d tokens which goes ouside the sequence."
            % (pos, nbOverlappingToken)
        )

        # Normalize candidates weights if needed
        normCandidates = {}

        # Limit the number of candiates
        for k, v in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True):
            normCandidates[k] = v
            if len(normCandidates) >= maxNbCandidate:
                break

        if len(normCandidates) > 0 and sum(normCandidates.values()) != 1:
            print(
                "[WARNING] : Normalizing weights at %s:%s"
                % (pos_nbtokens, str(normCandidates))
            )
            normCandidates = {
                cor: float(x) / sum(normCandidates.values())
                for cor, x in normCandidates.items()
            }

        tokenPosErrReshaped[pos] = {
            "nbToken": nbOverlappingToken,
            "boundsAligned": boundsAligned,
            "ocrSeq": re.sub(
                evalContext.charExtend,
                "",
                evalContext.ocrAligned[
                    boundsAligned[0] : boundsAligned[0] + boundsAligned[1]
                ],
            ),
            "gsSeq": re.sub(
                evalContext.charExtend,
                "",
                evalContext.gsAligned[
                    boundsAligned[0] : boundsAligned[0] + boundsAligned[1]
                ],
            ),
            "candidates": normCandidates,
        }

    # Debug test
    if verbose:
        EvalContext.printDicoSortedByKey(tokenPosErrReshaped, "tokenPosErrReshaped")

    return tokenPosErrReshaped

# %% ../nbs/03_utils.ipynb 59
def runEvaluation(
    datasetDirPath,  # path to the dataset directory (ex: r"./dataset_sample")
    pathInputJsonErrorsCorrections,  # # input path to the JSON result (ex: r"./inputErrCor_sample.json"), format given on https://sites.google.com/view/icdar2017-postcorrectionocr/evaluation)
    pathOutputCsv,  # output path to the CSV evaluation results (ex: r"./outputEval.csv")
    verbose=False,
):  # Show verbose output
    """Main evaluation method"""

    # Load results JSON file
    with codecs.open(
        pathInputJsonErrorsCorrections, "r", encoding="utf-8"
    ) as data_file:
        formatedRes = json.loads(data_file.read())

    # CSV header fields
    csvHeader = [
        "File",
        "NbTokens",
        "NbErroneousTokens",
        "NbSymbolsConsidered",  # NbTokens furtherly used to weight file's metrics \
        "T1_Precision",
        "T1_Recall",
        "T1_Fmesure",  # Task 1) Metrics \
        "T2_AvgLVDistOriginal",
        "T2_AvgLVDistCorrected",
    ]  # Task 2) Metrics

    # Write CSV file's header into a new output file
    with open(pathOutputCsv, "w") as outputFile:
        outputFile.write(";".join(csvHeader) + "\n")

    # Print CSV header into the console file
    print("\t".join(csvHeader))

    # Iterate over all the file's paths given in the input results
    for filePath, tokenPosErr in formatedRes.items():
        # Load the context : [OCR_toInput], [OCR_aligned] and [ GS_aligned]
        evalContext = EvalContext(
            os.path.join(datasetDirPath, filePath), verbose=verbose
        )

        # Compute some intrinsic statistics
        nbTokens, nbErrTokens, nbErrTokensAlpha = evalContext.get_errors_stats()

        tokenPosErrReshaped = reshape_input_errors(tokenPosErr, evalContext, verbose)

        # Task 1) Run the evaluation : Detection of the position of erroneous tokens
        prec, recall, fmes = evalContext.task1_eval(tokenPosErrReshaped)

        # Task 2) Run the evaluation : Correction of the erroneous tokens
        (
            sumCorrectedDistance,
            sumOriginalDistance,
            nbSymbolsConsidered,
        ) = evalContext.task2_eval(tokenPosErrReshaped, useFirstCandidateOnly=False)

        # Manage division per zero
        avgCorrectedDistance = (
            sumCorrectedDistance / float(nbSymbolsConsidered)
            if nbSymbolsConsidered > 0
            else 0
        )
        avgOriginalDistance = (
            sumOriginalDistance / float(nbSymbolsConsidered)
            if nbSymbolsConsidered > 0
            else 0
        )

        # Format results in CSV
        strRes = "%s;%d;%d;%d;%0.02f;%0.02f;%0.02f;%0.02f;%0.02f" % (
            filePath,
            nbTokens,
            nbErrTokens,
            nbSymbolsConsidered,
            prec,
            recall,
            fmes,
            avgOriginalDistance,
            avgCorrectedDistance,
        )

        # Write results in output file
        with open(pathOutputCsv, "a") as outputFile:
            outputFile.write(strRes + "\n")

        # Print results in the console
        print(strRes.replace(";", "\t"))

# %% ../nbs/03_utils.ipynb 60
def read_results(csv_file):
    """Read csv with evaluation results"""
    data = pd.read_csv(csv_file, sep=";")
    data["language"] = data.File.apply(lambda x: x[:2])
    data["subset"] = data.File.apply(lambda x: x.split("/")[1])

    return data

# %% ../nbs/03_utils.ipynb 62
def icdar_output2simple_correction_dataset_df(
    output: Dict[str, Dict[str, Dict]], data: Dict[str, Text], dataset: str = "test"
) -> pd.DataFrame:
    """Convert the icdar data error detection output to input for SimpleCorrectionDataset

    Because gold standard for input_tokens is not available, the dataset dataframe cannot
    be used for evaluation anymore.
    """
    samples = []
    for key, mistakes in output.items():
        text = data[key]
        for token in mistakes:
            sample = {}
            parts = token.split(":")
            start_idx = int(parts[0])
            num_tokens = int(parts[1])
            for i, at in enumerate(text.input_tokens):
                if at.start == start_idx:
                    sample['key'] = key
                    sample["ocr"] = " ".join([t.ocr for t in text.input_tokens[i: i+num_tokens]])
                    sample["gs"] = " ".join([t.gs for t in text.input_tokens[i: i+num_tokens]]).strip()
                    sample["start"] = at.start
                    sample["text"] = key
                    sample["token"] = token
                    sample["len_ocr"] = len(sample["ocr"])
                    sample["len_gs"] = len(sample["gs"])
                    parts = key.split("/")
                    sample["language"] = parts[0]
                    sample["subset"] = parts[1]
                    sample["dataset"] = dataset

            if sample == {}:
                raise ValueError(f"No token found for {key}, start index: {start_idx}")
            samples.append(sample)
    return pd.DataFrame(samples)

# %% ../nbs/03_utils.ipynb 66
def read_results(csv_file):
    data = pd.read_csv(csv_file, sep=';')
    data['language'] = data.File.apply(lambda x: x[:2])
    data['subset'] = data.File.apply(lambda x: x.split('/')[1])

    return data


def aggregate_results(csv_file):
    data = read_results(csv_file)

    return data.groupby("language").mean()[["T1_Precision", "T1_Recall", "T1_Fmesure"]]


def aggregate_ed_results(csv_file):
    data = read_results(csv_file)

    data['ed_diff'] = data['T2_AvgLVDistOriginal'] - data['T2_AvgLVDistCorrected']
    data['%ed_improvement'] = data['ed_diff'] / data['T2_AvgLVDistOriginal'] * 100

    # If `T2_AvgLVDistOriginal` == 0.0 and `T2_AvgLVDistCorrected` > 0, `ed_diff` is a
    # negative number and `ed_diff`/`T2_AvgLVDistOriginal` = -inf.
    # The mean of numbers which include -inf is nan.
    # Therefore, -inf is replaced by -100.0%.
    data['%ed_improvement'].replace([-np.inf], -100.0, inplace=True)

    # If `T2_AvgLVDistOriginal` == 0.0 and and `T2_AvgLVDistCorrected` == 0.0,
    # the % improvement is nan. The mean of numbers which include nan is nan.
    # So, in this case, the value should be replaced with 0.0.
    data['%ed_improvement'].fillna(0.0, inplace=True)

    return data.groupby("language").mean()[['%ed_improvement']]

# %% ../nbs/03_utils.ipynb 70
def reduce_dataset(dataset, n=5):
    """Return dataset with the first n samples for each split"""
    for split in dataset.keys():
        dataset[split] = dataset[split].select(range(n))
    return dataset
