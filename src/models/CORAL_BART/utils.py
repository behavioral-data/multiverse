import json
import numpy as np
from collections import defaultdict
from itertools import chain
import networkx as nx
import os
import random
import socket


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_decode(tokenizer, token_ids):
    token_ids = [x for x in token_ids if not x == -100]
    result = tokenizer.decode(token_ids, skip_special_tokens=True)
    if len(result) == 0 or all([x == "." for x in result]):
        return "<EMPTY>"
    return result


def safe_decode_coral(tokenizer, token_ids, idx2word):
    token_ids = [x for x in token_ids if (not x == -100) and (not x == 1)]

    result = ''.join([idx2word[i] for i in token_ids])
    if len(result) == 0 or all([x == "." for x in result]):
        return "<EMPTY>"
    return result


def clean_datum_for_serialization(datum):
    for k, v in datum.items():
        if isinstance(v, (np.ndarray, np.generic)):
            datum[k] = v.tolist()
    return datum


def write_jsonl(open_file, data, mode="a"):
    for datum in data:
        clean_datum = clean_datum_for_serialization(datum)
        open_file.write(json.dumps(clean_datum))
        open_file.write("\n")


def load_pickled_tree(path):
    if os.path.isdir(path):
        return nx.read_gpickle(os.path.join(path, "tree.pickle"))
    else:
        return nx.read_gpickle(path)


def block_shuffle(items, key_fn, return_blocks=False,seed=None):
    key_lists = defaultdict(list)
    for item in items:
        key = key_fn(item)
        key_lists[key].append(item)
    blocks = list(key_lists.values())
    random.Random(seed).shuffle(blocks)

    if return_blocks:
        return blocks

    result = list(chain.from_iterable(blocks))
    return result


def has_internet():
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False


def separate_regions(a, m):
    m0 = np.concatenate(([False], m, [False]))
    idx = np.flatnonzero(m0[1:] != m0[:-1])
    return [a[idx[i]:idx[i + 1]] for i in range(0, len(idx), 2)]
