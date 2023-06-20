import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import copy
from tqdm import tqdm
import random
import re

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

def Preprocessing(Tokenizer, FilePath, MaxTokenLen=512, Seed=42):
    UNCASE = True
    FileObject = open("./data/DefinitionDataset.txt", 'r')
    VocabDefExamDict = {}
    MAX_SEQ_LEN = 191
    pbar = tqdm(total = 1)
    for line in FileObject:
        Contents = line.split(" (def.) ")
        Word = Contents[0].strip()
        if UNCASE: Word = Word.lower()
        ### Word -> Definition1 -> [ex1, ex2]
        Defs = {}
        for c in Contents[1:]:
            c = c.split(" (ex.)")
            Def = c[0].strip(); Examples = c[1].strip()
            if UNCASE:
                Def = Def.lower()
                Examples = Examples.lower()

            Examples = Examples.split(' | ')

            ### When don't know MAX_SEQ_LEN
    #         SEQ_LEN =  max(*[len(Tokenizer.encode(e)) for e in Examples], len(Tokenizer.encode(Def)))
    #         if SEQ_LEN > MAX_SEQ_LEN:
    #             MAX_SEQ_LEN = SEQ_LEN

            Defs[Def] = Examples if len(Examples) else ''

        if Word in VocabDefExamDict.keys():
            VocabDefExamDict[Word].update(Defs)
        else:
            VocabDefExamDict[Word] = Defs
        pbar.update(1)
    pbar.close()

    MAX_SEQ_LEN = min(512, MAX_SEQ_LEN)
    MAX_SEQ_LEN = max(256, MAX_SEQ_LEN)

    print(len(VocabDefExamDict), MAX_SEQ_LEN)
    
    
    PAIR_list = ["Word-ExamWord","Def-ExamWord", "Word-Word", "Word-Def",]
#     PAIR_list = ["Word-Def",]
    PAIR_cnt = dict(zip(PAIR_list, [0]*len(PAIR_list)))
    # MAX_SEQ_LEN = 512

    WordDef_train_src, DefExam_train_src, WordExam_train_src, WordWord_train_src, WordExWord_train_src, DefExWord_train_src = [],[],[],[],[],[]
    WordDef_train_tgt, DefExam_train_tgt, WordExam_train_tgt, WordWord_train_tgt, WordExWord_train_tgt, DefExWord_train_tgt = [],[],[],[],[],[]
    # WordDict = 
    ### Saved Indices
    CLS_id = Tokenizer.convert_tokens_to_ids(["[CLS]"])
    SEP_id = Tokenizer.convert_tokens_to_ids(["[SEP]"])
    PAD_id = Tokenizer.convert_tokens_to_ids(["[PAD]"])
    SpTokenCnt = 2 # Except for PAD

    for PAIR in PAIR_list:
        pbar = tqdm(total=len(VocabDefExamDict))
        for i, Word in enumerate(VocabDefExamDict):
            pbar.update(1)
            Defs = list(VocabDefExamDict[Word].keys())
            if PAIR == "Word-Def":
#                 Word_ = Tokenizer.tokenize(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                for d in Defs:
#                     WordDef_train_src.append(CLS_id + Word + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(Word)))
#                     d = Tokenizer.encode(d, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                     WordDef_train_tgt.append(CLS_id + d + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(d)))
#                     PAIR_cnt[PAIR] += 1

#                     d = Tokenizer.tokenize(d, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                     WordDef_train_src.append(' '.join(["[CLS]"] + Word + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(Word))))
#                     WordDef_train_tgt.append(' '.join(["[CLS]"] + d + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(d))))

#                     d = Tokenizer.tokenize(d, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                    WordDef_train_src.append(f"Definition of {Word} is {d}")
                    WordDef_train_tgt.append(f"Definition of {Word} is {d}")
#                     WordDef_train_tgt.append(d)
                    PAIR_cnt[PAIR] += 1

            elif PAIR == "Def-Exam":
                w = VocabDefExamDict[Word]
                for d in Defs:
#                     d_ = Tokenizer.encode(d, add_special_tokens=False)[:MAX_SEQ_LEN-2]
#                     for ex in w[d]:
#                         if ex != '':
#                             DefExam_train_src.append(CLS_id + d_ + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(d_)))
#                             ex = Tokenizer.encode(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                             DefExam_train_tgt.append(CLS_id + ex + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(ex)))
                    d_ = Tokenizer.tokenize(d, add_special_tokens=False)[:MAX_SEQ_LEN-2]
                    for ex in w[d]:
                        if ex != '':
                            DefExam_train_src.append(' '.join(["[CLS]"] + d_ + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(d_))))
                            ex = Tokenizer.tokenize(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                            DefExam_train_tgt.append(' '.join(["[CLS]"] + ex + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(ex))))
                        PAIR_cnt[PAIR] += 1

            elif PAIR == "Word-Exam": # Rarely used
                w = VocabDefExamDict[Word]
#                 Word_ = Tokenizer.encode(Word, add_special_tokens=False)[:MAX_SEQ_LEN-2]
#                 for d in Defs:
#                     for ex in w[d]:
#                         if ex != '':
#                             WordExam_train_src.append(CLS_id + Word_ + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(Word_)))
#                             ex = Tokenizer.encode(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                             WordExam_train_tgt.append(CLS_id + ex + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(ex)))
#                             PAIR_cnt[PAIR] += 1
                Word_ = Tokenizer.tokenize(Word, add_special_tokens=False)[:MAX_SEQ_LEN-2]
                for d in Defs:
                    for ex in w[d]:
                        if ex != '':
                            WordExam_train_src.append(' '.join(["[CLS]"] + Word_ + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(Word_))))
                            ex = Tokenizer.tokenize(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                            WordExam_train_tgt.append(' '.join(["[CLS]"] + ex + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(ex))))
                            
                            PAIR_cnt[PAIR] += 1

            elif PAIR == "Word-Word":
#                 w = VocabDefExamDict[Word]
#                 Word_ = Tokenizer.encode(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                 WordWord_train_src.append(CLS_id + Word_ + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(Word_)))
#                 WordWord_train_tgt.append(CLS_id + Word_ + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(Word_)))
#                 PAIR_cnt[PAIR] += 1
                w = VocabDefExamDict[Word]
#                 Word_ = Tokenizer.tokenize(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                 WordWord_train_src.append(' '.join(["[CLS]"] + Word_ + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(Word_))))
#                 WordWord_train_tgt.append(' '.join(["[CLS]"] + Word_ + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(Word_))))

                WordWord_train_src.append(f"{Word} is {Word}")
                WordWord_train_tgt.append(f"{Word} is {Word}")
                PAIR_cnt[PAIR] += 1

            elif PAIR == "Word-ExamWord":
                w = VocabDefExamDict[Word]
#                 Word_ = Tokenizer.encode(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                 for d in Defs:
#                     for ex in w[d]:
#                         if ex != '':
#                             WordExWord_train_src.append(CLS_id + Word_ + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(Word_)))
#                             ex = Tokenizer.encode(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                             WordExWord_train_tgt.append(CLS_id + ex + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(ex)))
#                             PAIR_cnt[PAIR] += 1
                Word_ = Tokenizer.tokenize(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                for d in Defs:
                    for ex in w[d]:
                        if ex != '':
#                             WordExWord_train_src.append(' '.join(["[CLS]"] + Word_ + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(Word_))))
#                             ex = Tokenizer.tokenize(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                             WordExWord_train_tgt.append(' '.join(["[CLS]"] + ex + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(ex))))
                            WordExWord_train_src.append(f"Example of {Word} is {ex}")
                            WordExWord_train_tgt.append(f"Example of {Word} is {ex}")
                            PAIR_cnt[PAIR] += 1

            elif PAIR == "Def-ExamWord":
                w = VocabDefExamDict[Word]
#                 Word_ = Tokenizer.encode(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                 for d in Defs:
#                     d_ = Tokenizer.encode(d, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                     for ex in w[d]:
#                         if ex != '':
#                             DefExWord_train_src.append(Word_ + PAD_id*(10-len(Word_)) + CLS_id + d_ + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(d_)))
#                             ex = Tokenizer.encode(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                             DefExWord_train_tgt.append(CLS_id + ex + SEP_id + PAD_id*(MAX_SEQ_LEN-SpTokenCnt-len(ex)))
#                             PAIR_cnt[PAIR] += 1
#                 Word_ = Tokenizer.tokenize(Word, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                for d in Defs:
#                     d_ = Tokenizer.tokenize(d, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
                    for ex in w[d]:
                        if ex != '':
                            DefExWord_train_src.append(f"Example of Definition {d} is {ex}")
                            DefExWord_train_tgt.append(f"Example of Definition {d} is {ex}")
#                             DefExWord_train_src.append(' '.join(Word_ + ["[PAD]"]*(10-len(Word_)) + ["[CLS]"] + d_ + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(d_))))
#                             ex = Tokenizer.tokenize(ex, add_special_tokens=False)[:MAX_SEQ_LEN-SpTokenCnt]
#                             DefExWord_train_tgt.append(["[CLS]"] + ex + ["[SEP]"] + ["[PAD]"]*(MAX_SEQ_LEN-SpTokenCnt-len(ex)))
                            PAIR_cnt[PAIR] += 1
            else:
                print("Out-of-Bound")
                break

            ### For Debug
    #         break
    pbar.close()

    train_src, valid_src, test_src = [], [], []
    src = WordDef_train_src + DefExam_train_src + WordExam_train_src + WordWord_train_src + WordExWord_train_src + DefExWord_train_src
    train_tgt, valid_tgt, test_tgt = [], [], []
    tgt = WordDef_train_tgt + DefExam_train_tgt + WordExam_train_tgt + WordWord_train_tgt + WordExWord_train_tgt + DefExWord_train_tgt
#     if SplitMethod == "ByPaperID":
    if False:
        try:
#             raise
            FileObject = open("SplitInfo_valid.txt", 'r', encoding="utf-8")
            ValidPaperID = [line.strip() for line in FileObject]
            FileObject.close()
            FileObject = open("SplitInfo_test.txt", 'r', encoding="utf-8")
            TestPaperID = [line.strip() for line in FileObject]
            FileObject.close()
        except:
            print(int(len(set(d['paper_id'] for d in data))*0.10)) # 10%
            random.seed(42)
            ValidPaperID = random.choices(list(set(d['paper_id'] for d in data)),
                                          k=int(len(set(d['paper_id'] for d in data))*0.20)) # 20%
            TestPaperID = sorted(ValidPaperID[len(ValidPaperID)//2:]) # 10% for Test
            ValidPaperID = sorted(ValidPaperID[:len(ValidPaperID)//2]) # 10% for Validation

            FileObject = open("SplitInfo_valid.txt", 'w', encoding='utf-8')
            for ids in ValidPaperID:
                FileObject.write(ids+'\n')
            FileObject.close()
            FileObject = open("SplitInfo_test.txt", 'w', encoding='utf-8')
            for ids in TestPaperID:
                FileObject.write(ids+'\n')
            FileObject.close()
            
        train_src, valid_src, test_src = [], [], []
        train_tgt, valid_tgt, test_tgt = [], [], []
        for i, d in enumerate(data):
            if i >= len(src): break
#             if i in MissingIndex: continue
            if d['paper_id'] in ValidPaperID:
                valid_src.append(src[i])
                valid_tgt.append(tgt[i])
            elif d['paper_id'] in TestPaperID:
                test_src.append(src[i])
                test_tgt.append(tgt[i])
            else:
                train_src.append(src[i])
                train_tgt.append(tgt[i])

    elif True:
        tgt = src
        train_src = src[:int(len(src)*0.9)]
        valid_src = src[int(len(src)*0.9):int(len(src)*0.95)]
        test_src = src[int(len(src)*0.95):]
        train_tgt = tgt[:int(len(tgt)*0.9)]
        valid_tgt = tgt[int(len(tgt)*0.9):int(len(tgt)*0.95)]
        test_tgt = tgt[int(len(tgt)*0.95):]

    elif False:
        ValidIdx = random.uniform(0,0.95)
        TestIdx = random.uniform(0,0.95)
        while ValidIdx <= TestIdx <= ValidIdx+0.05 or TestIdx <= ValidIdx <= TestIdx+0.05:
            ValidIdx = random.uniform(0,0.95)
            TestIdx = random.uniform(0,0.95)
        train_src = src[:int(len(src)*min(ValidIdx, TestIdx))]
        train_src += src[int(len(src)*min(ValidIdx, TestIdx)+0.05):int(len(src)*max(ValidIdx, TestIdx))]
        train_src += src[int(len(src)*max(ValidIdx, TestIdx)+0.05):]
        valid_src = src[int(len(src)*ValidIdx):int(len(src)*(ValidIdx+0.05))]
        test_src = src[int(len(src)*TestIdx):int(len(src)*(TestIdx+0.05))]
        train_tgt = tgt[:int(len(tgt)*min(ValidIdx, TestIdx))]
        train_tgt += tgt[int(len(tgt)*min(ValidIdx, TestIdx)+0.05):int(len(tgt)*max(ValidIdx, TestIdx))]
        train_tgt += tgt[int(len(tgt)*max(ValidIdx, TestIdx)+0.05):]
        valid_tgt = tgt[int(len(tgt)*ValidIdx):int(len(tgt)*(ValidIdx+0.05))]
        test_tgt = tgt[int(len(tgt)*TestIdx):int(len(tgt)*(TestIdx+0.05))]
    
    return (train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt)
