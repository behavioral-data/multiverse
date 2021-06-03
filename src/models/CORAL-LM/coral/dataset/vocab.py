# coding=utf-8
# created by Ge Zhang, April 3, 2020
# vocab

import os
import json
from tqdm import tqdm


def get_vocab_from_counter(counter, min_occur, max_size):
    """
    sort vocab and truncate it based on frequency
    """
    selected_tokens = [w for w in counter if counter[w] >= min_occur]
    selected_tokens = list(
        sorted(selected_tokens, key=lambda x: -counter[x]))

    idx2word = selected_tokens
    idx2word = idx2word[:max_size]

    word2idx = {w: i for i, w in enumerate(idx2word)}
    return idx2word, word2idx


class AnnotationVocab(object):
    """
    vocab for Markdown
    """

    def __init__(self, graphs, min_occur=3, max_size=10000):
        super(AnnotationVocab, self).__init__()

        self.min_occur = min_occur
        self.max_size = max_size

        counter = {}
        for g in tqdm(graphs):
            header = g["annotation"][-1] if len(g["annotation"]) > 0 else ""
            token = [t.lower() for t in header.split() if t]
            for t in token:
                if t not in counter:
                    counter[t] = 0
                counter[t] += 1

        idx2word, word2idx = get_vocab_from_counter(
            counter, min_occur, max_size)

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class CodeVocab(object):
    """
    Vocab for Code
    """

    def __init__(self, graphs, use_sub_token=False, min_occur=3, max_size=10000):
        super(CodeVocab, self).__init__()

        self.use_sub_token = use_sub_token
        self.min_occur = min_occur
        self.max_size = max_size

        counter = {}
        for g in tqdm(graphs):
            if use_sub_token:
                for n in g["new_nodes"]:
                    if n not in counter:
                        counter[n] = 0
                    counter[n] += 1
            else:
                for n in g["nodes"]:
                    token = n["type"] if "value" not in n else n["value"]
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1

        idx2word, word2idx = get_vocab_from_counter(
            counter, min_occur, max_size)

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class UnitedVocab(object):
    """
    Vocab
    """
    pad_index = 0
    unk_index = 1
    mask_index = 2
    sos_index = 3
    sep_index = 4
    eos_index = 5

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None):
        super(UnitedVocab, self).__init__()
        self.pad_index = 0
        self.unk_index = 1
        self.mask_index = 2
        self.sos_index = 3
        self.sep_index = 4
        self.eos_index = 5

        self.use_sub_token = use_sub_token
        self.min_occur = min_occur

        if not os.path.exists(path):
            code_vocab = CodeVocab(
                graphs, use_sub_token=use_sub_token, min_occur=min_occur)
            annotation_vocab = AnnotationVocab(
                graphs, min_occur=min_occur)
            idx2word = list(
                set(code_vocab.idx2word + annotation_vocab.idx2word))

            idx2word = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]", "[EOS]"] + \
                idx2word
            word2idx = {w: i for i, w in enumerate(idx2word)}
            with open(path, 'w') as fout:
                json.dump({"idx2word": idx2word,
                           "word2idx": word2idx}, fout)

        else:
            with open(path, 'r') as f:
                dictionary = json.load(f)
                idx2word = dictionary["idx2word"]
                word2idx = dictionary["word2idx"]

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)
