## From https://git.univ-lr.fr/gchiro01/icdar2017/blob/master/evalTool_ICDAR2017.py on CC0 License
## Via https://github.com/Kotwic4/ocr-correction/blob/master/ocr_correction/dataset/icdar/evalTool_ICDAR2017.py

# -*- coding: utf-8 -*-
import itertools, re, os, json, sys, codecs, collections

# For debugging purpose
verbose = False
maxNbCandidate = 6

################# MAIN EVALUATION METHOD ################

# - datasetDirPath: path to the dataset directory (ex: r"./dataset_sample")
# - pathInputJsonErrorsCorrections: input path to the JSON result (ex: r"./inputErrCor_sample.json"), format given on https://sites.google.com/view/icdar2017-postcorrectionocr/evaluation)
# - pathOutputCsv: output path to the CSV evaluation results (ex: r"./outputEval.csv")

# See if __name__ == '__main__' below for shell usage...

def runEvaluation(datasetDirPath, pathInputJsonErrorsCorrections, pathOutputCsv):

    # Load results JSON file
    with codecs.open(pathInputJsonErrorsCorrections, "r", encoding="utf-8") as data_file:
        formatedRes = json.loads(data_file.read())

    # CSV header fields
    csvHeader = ["File", "NbTokens", "NbErroneousTokens", "NbSymbolsConsidered", # NbTokens furtherly used to weight file's metrics \
                 "T1_Precision", "T1_Recall", "T1_Fmesure", # Task 1) Metrics \
                 "T2_AvgLVDistOriginal", "T2_AvgLVDistCorrected"] # Task 2) Metrics

    # Write CSV file's header into a new output file
    with open(pathOutputCsv, 'w') as outputFile:
        outputFile.write(";".join(csvHeader) + "\n")

    # Print CSV header into the console file
    print("\t".join(csvHeader))

    # Iterate over all the file's paths given in the input results
    for filePath, tokenPosErr in formatedRes.items():

        # Load the context : [OCR_toInput], [OCR_aligned] and [ GS_aligned]
        evalContext = EvalContext(os.path.join(datasetDirPath, filePath))

        # Compute some intrinsic statistics
        nbTokens, nbErrTokens, nbErrTokensAlpha = evalContext.get_errors_stats()

        # Store tokens' positions in mem
        tokensPos = [0] + [spacePos.start() + 1 for spacePos in re.finditer(r"\ ", evalContext.ocrOriginal)]

        # 1) Check JSON result format (ex: positions correspond to tokens)
        # 2) Reshape data "pos":{"nbTokens":..., "boundsAligned":..., candidates:... }"
        # 3) Locally normalize candidates' weights id needed
        tokenPosErrReshaped = {}
        for pos_nbtokens, candidates in tokenPosErr.items():

            pos, nbOverlappingToken = [int(i) for i in pos_nbtokens.split(':')]
            boundsAligned = evalContext.get_aligned_token_bounds(pos, nbOverlappingToken)

            # Check pos targets a existing token (first char)
            assert pos in tokensPos,\
                "[ERROR] : Error at pos %s does not target the first char of a token (space separated sequences)." % pos

            assert evalContext.ocrOriginal[pos:].count(" ") >= nbOverlappingToken-1,\
                "[ERROR] : Error at pos %d spreads overs %d tokens which goes ouside the sequence." % (pos,nbOverlappingToken)

            # Normalize candidates weights if needed
            normCandidates = {}

            # Limit the number of candiates
            for k,v in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True):
                normCandidates[k] = v
                if len(normCandidates) >= maxNbCandidate:
                    break

            if len(normCandidates) > 0 and sum(normCandidates.values()) != 1:
                print("[WARNING] : Normalizing weights at %s:%s" % (pos_nbtokens, str(normCandidates)) )
                normCandidates = {cor: float(x)/sum(normCandidates.values()) for cor, x in normCandidates.items()}

            tokenPosErrReshaped[pos] = {
                "nbToken":nbOverlappingToken,
                "boundsAligned":boundsAligned,
                "ocrSeq":re.sub(evalContext.charExtend, "",evalContext.ocrAligned[boundsAligned[0]:boundsAligned[0]+boundsAligned[1]]),
                "gsSeq":re.sub(evalContext.charExtend, "",evalContext.gsAligned[boundsAligned[0]:boundsAligned[0] + boundsAligned[1]]),
                "candidates":normCandidates
            }

        # Debug test
        if verbose:
            EvalContext.printDicoSortedByKey(tokenPosErrReshaped, "tokenPosErrReshaped")


        # Task 1) Run the evaluation : Detection of the position of erroneous tokens
        prec, recall, fmes = evalContext.task1_eval(tokenPosErrReshaped)

        # Task 2) Run the evaluation : Correction of the erroneous tokens
        sumCorrectedDistance, sumOriginalDistance, nbSymbolsConsidered = evalContext.task2_eval(tokenPosErrReshaped, useFirstCandidateOnly=False)

        # Manage division per zero
        avgCorrectedDistance = sumCorrectedDistance / float(nbSymbolsConsidered) if nbSymbolsConsidered > 0 else 0
        avgOriginalDistance = sumOriginalDistance / float(nbSymbolsConsidered) if nbSymbolsConsidered > 0 else 0

        # Format results in CSV
        strRes = "%s;%d;%d;%d;%0.02f;%0.02f;%0.02f;%0.02f;%0.02f" % \
                (filePath, nbTokens, nbErrTokens, nbSymbolsConsidered, prec, recall, fmes, avgOriginalDistance, avgCorrectedDistance)

        # Write results in output file
        with open(pathOutputCsv, 'a') as outputFile:
            outputFile.write(strRes + "\n")

        # Print results in the console
        print(strRes.replace(";", "\t"))



################# CLASS FOR STORING CURRENT FILE CONTEXT  ################
class EvalContext:

    # Default symbols used for the alignment and for ignoring some tokens
    charExtend = r"@"
    charIgnore = r"#"

    # Different texts versions provided
    ocrAligned, gsAligned, ocrOriginal = "","",""

    # Alignment map (ocrOriginal <=> ocrAligned and gsAligned)
    aMap = []

    def __init__(self, filePath):

        assert os.path.exists(filePath), "[ERROR] : File %s not found !" % filePath

        self.filePath = filePath

        # Load file data
        with open(filePath, 'r') as f:
            text = f.read().strip()
            self.ocrOriginal, self.ocrAligned, self.gsAligned = [txt[14:] for txt in re.split(r"\r?\n", text)]

            if self.charExtend in self.ocrOriginal:
                print(f'{self.charExtend} found in ocrOriginal. Removing...')
                self.ocrOriginal = self.ocrOriginal.replace(self.charExtend, '')

        # Check file integrity
        assert self.ocrOriginal == re.sub(self.charExtend, "", self.ocrAligned), "[ERROR] : [OCR_aligned] without \"%s\" doesn't correspond to [OCR_toInput] " % self.charExtend

        # Build the alignment map
        self.aMap = [x.start() - i for i, x in enumerate(re.finditer(self.charExtend + r"|$", self.ocrAligned))]

        #print("%s\n%s\n%s" % (self.ocrOriginal, self.ocrAligned, self.gsAligned))

    # Get the alignment shift for a position in the orginal OCR to the corresponding postion in the aligned OCR/GS
    def get_aligned_shift(self, posOriginal):
        return next((i for i, e in enumerate(self.aMap) if e >= posOriginal), 0)

    # Get the alignment shift for a position in the orginal OCR to the corresponding postion in the aligned OCR/GS
    def get_original_shift(self, posAligned):
        return self.ocrAligned.count(self.charExtend, 0, posAligned)

    # Get bounds in "Aligned OCR/GS" from a token position in the non-aligned OCR.
    def get_aligned_token_bounds(self, tokenPos, nbToken=1):

        assert (tokenPos == 0) or (self.ocrOriginal[tokenPos-1] == " "), \
            "[ERROR] : %d is not a token start position (%s)" % (tokenPos, self.filePath)

        alignedPos = tokenPos + self.get_aligned_shift(tokenPos)
        seqLen = nbToken-1 # Init with number of spaces
        iterOcrAlignedSpace = re.finditer(r"$|\ ", self.ocrAligned[alignedPos:])
        for nbt in range(nbToken):
            matchSpace = next(iterOcrAlignedSpace, None)
            if matchSpace is None:
                print("[WARNING] : At pos %d, could not iterate forward over tokens, end of the sequence reached" % tokenPos)
                break
            seqLen = matchSpace.start() # Look for last space before next token

        return alignedPos, seqLen


    # Get statistics (erroneous tokens' posisitions, corrections, ect...) on errors
    def get_errors_stats(self):

        #results = {}
        nbTokens, nbErrTokens, nbErrTokensAlpha = 0, 0, 0

        # Iterate over GS tokens
        lastTokenPos = 0
        for spacePos in re.finditer(r"$|\ ", self.gsAligned):

            tokenEndPos = spacePos.start()
            tokenInOcr = re.sub(self.charExtend, "", self.ocrAligned[lastTokenPos:tokenEndPos])
            tokenInGs = re.sub(self.charExtend, "", self.gsAligned[lastTokenPos:tokenEndPos])

            if self.charIgnore in tokenInGs:
                lastTokenPos = tokenEndPos + 1
                continue

            #if (tokenInOcr != tokenInGs):
            #    results[lastTokenPos] = [tokenInOcr, tokenInGs]
            #    #print("_%s_%s_" % (tokenInOcr, tokenInGs))

            lastTokenPos = tokenEndPos + 1

            nbTokens += 1
            nbErrTokens += (tokenInOcr != tokenInGs)
            nbErrTokensAlpha += (tokenInOcr != tokenInGs) and tokenInGs.strip().isalpha()

        return nbTokens, nbErrTokens, nbErrTokensAlpha



    def collectErrorPos(self):

        errorList = []

        # Add tolerance to hyphens : work on a new GS where hyphens founds in OCR are considered to be ignored in task1
        gsAlignedHyphensIgnored = self.gsAligned
        for hToken in re.finditer(r"[^ ]*((\ ?-[^ ])|([^ ]-\ ?))[^ ]*", self.ocrAligned):
            gsAlignedHyphensIgnored = gsAlignedHyphensIgnored[:hToken.start()] + self.charIgnore*(hToken.end()-hToken.start()) + gsAlignedHyphensIgnored[hToken.end():]

        gsSpacePos = set([spacePos.start() for spacePos in re.finditer(r"^|$|\ ", gsAlignedHyphensIgnored)])
        ocrSpacePos = set([spacePos.start() for spacePos in re.finditer(r"^|$|\ ", self.ocrAligned)])
        commonSpacePos = sorted(gsSpacePos.intersection(ocrSpacePos))
        #print(commonSpacePos)

        for i in range(len(commonSpacePos)-1):

            tokenStartPos = commonSpacePos[i] + 1*(gsAlignedHyphensIgnored[commonSpacePos[i]]==" ")
            tokenEndPos = commonSpacePos[i+1]

            tokenInGs = re.sub(self.charExtend, "", gsAlignedHyphensIgnored[tokenStartPos:tokenEndPos])
            tokenInOcr = re.sub(self.charExtend, "", self.ocrAligned[tokenStartPos:tokenEndPos])

            # Get not aligned pos
            tokenStartPosOriginal = tokenStartPos - self.get_original_shift(tokenStartPos)

            # Ignore the "#" in GS
            if not (self.charIgnore in tokenInGs):
                if (tokenInOcr != tokenInGs):
                    #print("%d:%d|%d=%s=>%s" % (tokenStartPosOriginal, tokenInOcr.count(" ")+1, tokenStartPos , tokenInOcr, tokenInGs))
                    errorList.append("%d:%d" % (tokenStartPosOriginal, tokenInOcr.count(" ")+1))
                    #errorList.append("%d:%d|%d=%s=>%s" % (tokenStartPosOriginal, tokenInOcr.count(" ") + 1, tokenStartPos, tokenInOcr, tokenInGs))

        return errorList


    def task1_eval(self, inputErroneousTokens):

        # Add tolerance to hyphens : work on a new GS where hyphens founds in OCR are considered to be ignored in task1
        gsAlignedHyphensIgnored = self.gsAligned
        for hToken in re.finditer(r"[^ ]*((\ ?-[^ ])|([^ ]-\ ?))[^ ]*", self.ocrAligned):
            gsAlignedHyphensIgnored = gsAlignedHyphensIgnored[:hToken.start()] + self.charIgnore*(hToken.end()-hToken.start()) + gsAlignedHyphensIgnored[hToken.end():]

        # 1) Prepare input results : unfold overlapping n tokens in tokenPosErr
        detectedErrPosUnfolded = {}
        for errPos, val in inputErroneousTokens.items():
            tokenStartPos = 0
            iterTokens = re.finditer(r"$|\ ", self.ocrOriginal[errPos:])
            for t in range(val["nbToken"]):

                alignedPos, seqLen = self.get_aligned_token_bounds((errPos + tokenStartPos), 1)
                rawTokenInAlignedGs = gsAlignedHyphensIgnored[alignedPos:(alignedPos + seqLen + 1)]

                if not (self.charIgnore in rawTokenInAlignedGs):

                    # Check if there is no overlapping errors/corrections
                    assert alignedPos not in detectedErrPosUnfolded, "[ERROR] : Error at pos %d is overlapping another given error ! Pay attention to the number of overlapping tokens." % (errPos + tokenStartPos)
                    detectedErrPosUnfolded[alignedPos] = [rawTokenInAlignedGs, val["candidates"]]

                tokenEndMatch = next(iterTokens, None)
                if tokenEndMatch is None:
                    break

                tokenStartPos = tokenEndMatch.start() + 1

        if verbose:
            EvalContext.printDicoSortedByKey(detectedErrPosUnfolded, "1) detectedErrPosUnfolded")


        # 2) Prepare real results
        realErrPosUnfolded = {}
        iterTokens = re.finditer(r"$|\ ", self.ocrAligned)
        tokenStartPos = 0
        tokenEndMatch = next(iterTokens, None)

        while tokenEndMatch is not None:

            tokenEndPos = tokenEndMatch.start() + 1  # Include following char

            tokenInGs = re.sub(self.charExtend, "", gsAlignedHyphensIgnored[max(0,tokenStartPos-1):tokenEndPos])
            tokenInOcr = re.sub(self.charExtend, "", self.ocrAligned[max(0,tokenStartPos-1):tokenEndPos])

            # Ignore the "#" in GS
            if not (self.charIgnore in tokenInGs):
                if (tokenInOcr != tokenInGs):
                    realErrPosUnfolded[tokenStartPos] = [tokenInOcr, tokenInGs]

            tokenStartPos = tokenEndPos
            tokenEndMatch = next(iterTokens, None)

        if verbose:
            EvalContext.printDicoSortedByKey(realErrPosUnfolded, "2) realErrPosUnfolded")

        realErrPos = set(realErrPosUnfolded.keys())

        # 3) Compute classical metrics
        setErrPos = set(detectedErrPosUnfolded.keys())
        errTP = len(setErrPos.intersection(realErrPos))  # TruePositive
        errFP = len(setErrPos.difference(realErrPos))  # TrueNegative
        errFN = len(realErrPos.difference(setErrPos))  # FalseNegative


        # Possible division per 0
        prec = (errTP / float(errTP + errFP)) if (errTP + errFP) > 0 else 0
        recall = (errTP / float(errTP + errFN)) if (errTP + errFN) > 0 else 0
        fmes = (2.0 * float(prec * recall) / float(prec + recall)) if (prec + recall) > 0 else 0

		# Debug test
        if verbose:
            print("TASK 1) ErrTP %d / errFP %d / errFN %d /" % (errTP, errFP, errFN) + \
                  " prec %0.2f / recall %0.2f / fmes %0.2f" % (prec, recall, fmes))

        return prec, recall, fmes


    def task2_eval(self, inputErroneousTokens, useFirstCandidateOnly=False):

        # Init list of tokens' levenshtein distances
        originalDistance, correctedDistance = [0], [0]
        nbSymbolsConsidered = 0

        # Get tokens (or sequences including hyphens)
        splitRegExp = r"(?=([^-]\ [^-]))" # Now supporting overlapping
        spaceOCR = set([sp.start()+1 for sp in re.finditer(splitRegExp, self.ocrAligned)])
        spaceGS  = set([sp.start()+1 for sp in re.finditer(splitRegExp, self.gsAligned)])


        # Collect running tokens defined in correction...
        spaceCorToRemove = []
        inputErroneousTokensAligned = {}
        for p, details in inputErroneousTokens.items():
            posAligned = p + self.get_aligned_shift(p)
            inputErroneousTokensAligned[posAligned] =  details
            for m in itertools.islice(re.finditer(r"$|\ ", self.ocrAligned[posAligned:]), details["nbToken"]-1):
                spaceCorToRemove.append(m.start() + posAligned)


        #print("spaceCommon %s" % str(sorted(spaceOCR.intersection(spaceGS))))
        #print("spaceCorToRemove %s" % str(sorted(spaceCorToRemove)))

        spaceCommon = (spaceOCR.intersection(spaceGS)).difference(spaceCorToRemove)
        spaceCommon.add(len(self.ocrAligned))

        # Iterate over comparable sequences
        lastTokenStartPos = 0
        for tokenEndPos in sorted(spaceCommon):

            tokenInGs = self.gsAligned[lastTokenStartPos:tokenEndPos]
            tokenInOcr = self.ocrAligned[lastTokenStartPos:tokenEndPos]

            # Get corrections concerned by this sequence :
            tokensPosToCorrect = set(range(lastTokenStartPos, tokenEndPos)).intersection(inputErroneousTokensAligned.keys())

            listCombinaisons = []
            for p in tokensPosToCorrect:
                listCombinaisons.append([[p, c] for c, w in inputErroneousTokensAligned[p]["candidates"].items()])

            tokenProposed = {}
            for combi in itertools.product(*iter(listCombinaisons)):

                # Default
                corToken = list(tokenInOcr)

                prodWeight = 1.0
                for pc in sorted(combi, key=lambda k: k[0], reverse=True):

                    offsetStart = inputErroneousTokensAligned[pc[0]]["boundsAligned"][0] - lastTokenStartPos
                    offsetEnd = offsetStart + inputErroneousTokensAligned[pc[0]]["boundsAligned"][1]

                    corToken[offsetStart:offsetEnd] = pc[1]
                    prodWeight = prodWeight * inputErroneousTokensAligned[pc[0]]["candidates"][pc[1]]

                tokenProposed[re.sub(self.charExtend, "", "".join(corToken))] = prodWeight

            # Fix GC 15/06/2017
            if len(tokenProposed) == 0:
                tokenProposed = {tokenInOcr:1.0}

            # In case we want to consider only the highest candidate
            if useFirstCandidateOnly and len(tokenProposed)>0:
                tokenProposed = {max(tokenProposed, key=tokenProposed.get): 1.0}

            # Consider results only if no "#" found in the token (or tokens sequence).
            if not (self.charIgnore in tokenInGs):

                # Update damerau_levenshtein_distance distances' lists
                ignoreList = [" -", "- ", "-", self.charExtend]
                originalDistance.append(EvalContext.damerau_levenshtein(tokenInOcr, tokenInGs, ignoreList=ignoreList))

                weightedSum = sum([EvalContext.damerau_levenshtein(token, tokenInGs, ignoreList=ignoreList) * float(w) for token, w in tokenProposed.items()])
                correctedDistance.append(weightedSum)
                nbSymbolsConsidered += len(tokenInOcr)

            else:
                # print("[IGNORED] Token _%s_%s_ => %s" % (tokenInOcr, tokenInGs, tokenProposed))
                pass

            #print("(%d=>%d) %s | %s | %s " % (lastTokenStartPos,tokenEndPos, re.sub(self.charExtend, "", tokenInOcr), re.sub(self.charExtend, "", tokenInGs), str(tokenProposed) ))
            lastTokenStartPos = tokenEndPos+1


        if verbose:
            print("TASK 2) correctedDistance %d vs originalDistance %d" % (sum(correctedDistance), sum(originalDistance)))

        #print(correctedDistance)

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

        if s1==s2:
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

                score[i + 1][j + 1] = min(score[i][j] + cost,
                                          score[i + 1][j] + 1,
                                          score[i][j + 1] + 1,
                                          score[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1))
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



# ------------ Case of autonomous usage, process arguments -----------
if __name__ == '__main__':

    # Read arguments
    if len(sys.argv) == 4:
        datasetDirPath, pathInputJsonErrorsCorrections, pathOutputCsv = sys.argv[1:4]
        print("Using:\ndatasetDirPath: %s \npathInputJsonErrorsCorrections: %s \npathOutputCsv: %s\n" % (
        datasetDirPath, pathInputJsonErrorsCorrections, pathOutputCsv))
    else:
        print(
            "Usage : python evalTools_ICDAR2017.py \"datasetDirPath\" \"pathInputJsonErrorCorrection\" \"pathOutputCsv\"")
        print(
            "Example : python evalTools_ICDAR2017.py \"./dataset_sample\" \"./inputErrCor_sample.json\" \"./output_eval.csv\"")
        exit()

    # Run the eval function
    runEvaluation(datasetDirPath, pathInputJsonErrorsCorrections, pathOutputCsv)