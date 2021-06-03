
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, SGD

from transformers import PreTrainedTokenizer, DataCollator, DataCollatorForLanguageModeling

import nbformat
# from parse_python import parse_snippet
from typing import Any, Callable, Dict, List, NewType, Tuple
InputDataClass = NewType("InputDataClass", Any)

from dataclasses import dataclass

import json
from glob import glob
import os
import ast
from tqdm import tqdm
import random
import itertools
import sys
import pdb
from src.models.CORAL_BART.token_edges import CodeTree

from diff_match_patch import diff_match_patch

class dmp_with_pprint(diff_match_patch):
    def __init__(self, *args, **kwargs):
        self.pp_removed_start_token = kwargs.pop(
            "removed_start_token", "<REMOVED>")
        self.pp_removed_end_token = kwargs.pop(
            "removed_end_token", "</REMOVED>")
        self.pp_inserted_start_token = kwargs.pop(
            "inserted_start_token", "<INSERTED>")
        self.pp_inserted_end_token = kwargs.pop(
            "inserted_end_token", "</INSERTED>")

        super().__init__(*args, **kwargs)

    def pprint(self, diffs):
        pretty_diff = []
        for (op, text) in diffs:
            if op == self.DIFF_INSERT:
                pretty_diff.append(f"{self.pp_inserted_start_token}{text}{self.pp_inserted_end_token}")
            elif op == self.DIFF_DELETE:
                pretty_diff.append(f"{self.pp_removed_start_token}{text}{self.pp_removed_end_token}")
            elif op == self.DIFF_EQUAL:
                pretty_diff.append(text)
        return "".join(pretty_diff)

    def pprint_separate(self, diffs):
        orig = []
        new = []

        for (op, text) in diffs:
            if op == self.DIFF_INSERT:
                new.append(f"{self.pp_inserted_start_token}{text}{self.pp_inserted_end_token}")
            elif op == self.DIFF_DELETE:
                orig.append(f"{self.pp_removed_start_token}{text}{self.pp_removed_end_token}")
            elif op == self.DIFF_EQUAL:
                new.append(text)
                orig.append(text)
        return "".join(orig), "".join(new)


def rpad_list(l, size, padding_value):
    return l + [padding_value] * (size - len(l))


def rpad_mat(m, size, padding_value):
    m = torch.cat((m, torch.zeros((m.shape[0], size - m.shape[0]))), dim=1)
    m = torch.cat((m, torch.zeros((size - m.shape[0], size))), dim=0)
    m[0, :] = 1
    m[:, 0] = 1
    return m
    # pdb.set_trace()
    # raise NotImplementedError


def remove_special_tokens(input, tokenizer):
    for spec_tok in tokenizer.all_special_tokens:
        input = input.replace(spec_tok, "")
    return input


@dataclass
class DynamicPaddingCollatorSeq2Seq:
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        first = features[0]
        batch = {}
        max_size = 0
        for k, v in first.items():
            max_size_v = max([len(x[k]) for x in features])
            max_size = max(max_size_v, max_size)

        for k, v in first.items():
            dtype = torch.long if isinstance(v[0], int) else torch.float

            if k in ["labels", "input_labels"]:
                padding_value = -100  # Default value to ignore in loss

            elif k in ["attention_mask", "loss_mask", "adj_mat"]:
                padding_value = 0
            else:
                padding_value = self.tokenizer.pad_token_id

            if k == "adj_mat":
                batch[k] = torch.stack([torch.tensor(rpad_mat(
                    f[k], max_size, padding_value=padding_value), dtype=dtype) for f in features])

            else:
                batch[k] = torch.stack([torch.tensor(rpad_list(
                    f[k], max_size, padding_value=padding_value), dtype=dtype) for f in features])

        return batch


@dataclass
class TypeDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """docstring for TypeDataCollatorForLanguageModeling"""

    def __call__(self, examples):
        batch_input_ids = [i["input_ids"] for i in examples]
        batch_type = [i["type"] for i in examples]
        batch_input_ids = self._tensorize_batch(batch_input_ids)
        batch_type = self._tensorize_batch(batch_type)
        if self.mlm:
            inputs, labels, batch_type, special_tokens_mask, indices_random, indices_replaced, masked_indices, probability_matrix = self.mask_tokens(
                batch_input_ids, batch_type)
            return {"input_ids": inputs,
                    "labels": labels,
                    "types": batch_type,
                    "special_tokens_mask":special_tokens_mask,
                    "indices_random":indices_random,
                    "indices_replaced":indices_replaced,
                    "masked_indices":masked_indices,
                    "probability_matrix":probability_matrix
            }
        else:
            raise NotImplementedError
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def mask_tokens(self, batch_input_ids: torch.Tensor, batch_type: torch.Tensor):
        """
        Prepare masked tokens batch_input_ids/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # pdb.set_trace()
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = batch_input_ids.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(
            special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool() & ~torch.tensor(special_tokens_mask, dtype=torch.bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        batch_type[~masked_indices] = -1
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        batch_input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        batch_input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return batch_input_ids, labels, batch_type, special_tokens_mask, indices_random, indices_replaced, masked_indices, probability_matrix


@dataclass
class DynamicPaddingCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        first = features[0]
        batch = {}

        for k, v in first.items():
            try:
                dtype = torch.long if isinstance(v[0], int) else torch.float
            except TypeError:
                dtype = torch.long if isinstance(v, int) else torch.float
            if k in ["loss_mask"]:
                batch[k] = [f[k] for f in features]
            elif isinstance(v, torch.Tensor):
                batch[k] = pad_sequence(
                    [f[k] for f in features], padding_value=self.tokenizer.pad_token_id, batch_first=True)
            else:
                batch[k] = pad_sequence([torch.tensor(f[k], dtype=dtype) for f in features],
                                        padding_value=self.tokenizer.pad_token_id, batch_first=True)
        return batch


class KaggleDiffsReader():
    def __init__(self, diff_path, max_size=None):
        self.diff_path = diff_path
        self.diffs = []

        with open(self.diff_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Diffs"):
                diff_line = json.loads(line)
                orig, new = self.split_orig_and_new(diff_line)
                inserted, removed = self.get_inserted_and_removed(diff_line)
                remained = self.get_remained(diff_line)

                diff = {
                    "metadata": diff_line["metadata"],
                    "orig": orig,
                    "new": new,
                    "diff": diff_line["cell_diff"],
                    "inserted": inserted,
                    "removed": removed,
                    "remained": remained
                }
                self.diffs.append(diff)
                if max_size != -1 and len(self.diffs) == max_size:
                    # pdb.set_trace()
                    break

    def __len__(self):
        return len(self.diffs)

    def __getitem__(self, i):
        return self.diffs[i]

    def is_indented(self,line):
        return not len(line) == len(line.lstrip())

    def try_to_fix_indents(self,lines):
        if not any([x[-1]==":" for x  in lines]):
            lines = [x.lstrip() for x in lines]
        return lines



    def remove_git_chars(self, lines):
        if len(lines) == 0:
            return []
        new_lines = []
        for line in lines:
            if line[0] in ["+","-"]:
                new_lines.append(" " + line[1:])
            else:
                new_lines.append(line)
        if any([x[0]==" " and x[1]!=" " for x in new_lines]):
            new_lines = [x[1:] if x[0]==" " else x for x in new_lines]

        # Attempt to left-justify the lines
        indents = [len(x) - len(x.lstrip()) for x in new_lines]
        min_indent = min(indents)
        new_indents = [" "*(x-min_indent) for x in indents]
        new_lines = [ind+x for ind,x in zip(new_indents,[x.lstrip() for x in new_lines])]

        #Try to fix indents
        new_lines = self.try_to_fix_indents(new_lines)

        return new_lines

    def get_inserted_and_removed(self, diff):
        lines = diff["cell_diff"].split("\n")
        inserted = self.remove_git_chars(
            [x for x in lines if len(x.strip()) > 0 and x[0] == "+"])
        removed = self.remove_git_chars(
            [x for x in lines if len(x.strip()) > 0 and x[0] == "-"])
        return inserted, removed

    def split_orig_and_new(self, diff):
        lines = diff["cell_diff"].split("\n")
        orig = self.remove_git_chars([x for x in lines if len(x.strip())>0 and x[0] != "+" ])
        new = self.remove_git_chars([x for x in lines if len(x.strip())>0 and x[0] != "-"])
        return  "\n".join(orig), "\n".join(new)

    def get_remained(self,diff):
        lines = diff["cell_diff"].split("\n")
        remained = self.remove_git_chars([x for x in lines if len(x.strip())>0 and not x[0] in ["+","-"]])
        return remained


class CoralDiffsReader(KaggleDiffsReader):
    """Diffs Reader for CORAL encoder"""

    def __init__(self, diff_path, max_size=None):
        # TODO:
        # - add tree for orig and new
        super(CoralDiffsReader, self).__init__(diff_path, max_size)
        parsable_diffs = []
        odd_diffs = []
        for d in tqdm(self.diffs):
            orig = d["orig"]
            new = d["new"]
            try:
                ast.parse(orig)
                ast.parse(new)
                parsable_diffs.append(d)
            except Exception as e:
                odd_diffs.append(d)
                pass
        print('Remove {} unparsable diffs'.format(len(odd_diffs)))
        self.diffs = parsable_diffs

        for d in tqdm(self.diffs):
            d["orig_tree"] = CodeTree(d["orig"])
            d["new_tree"] = CodeTree(d["new"])

        self.bos_token = "<s>"
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.idx2word = [self.bos_token, self.pad_token, self.eos_token,
                         self.unk_token, self.mask_token]
        self.word2idx = {self.bos_token: 0, self.pad_token: 1,
                         self.eos_token: 2, self.unk_token: 3, self.mask_token: 4}

        for d in tqdm(self.diffs, desc = "Building Vocab..."):
            for tk in d["orig_tree"].tokens:
                if tk.string not in self.word2idx:
                    self.word2idx[tk.string] = len(self.idx2word)
                    self.idx2word.append(tk.string)
            for tk in d["new_tree"].tokens:
                if tk.string not in self.word2idx:
                    self.word2idx[tk.string] = len(self.idx2word)
                    self.idx2word.append(tk.string)

        # with open()


class UCSDNotebookReader():
    def __init__(self, data_path, preload_in_memory=False, max_size=None):
        self.data_path = data_path
        self.preload_in_memory = preload_in_memory
        self.notebook_paths = []
        self.notebooks = []

        for path in glob(os.path.join(self.data_path, "*.ipynb")):
            self.notebook_paths.append(path)
            if self.preload_in_memory:
                self.notebooks.append(nbformat.read(path, as_version=4))
            if max_size and len(self.notebooks) == max_size:
                break

    def __len__(self):
        return len(self.notebook_paths)

    def __getitem__(self, i):
        if self.preload_in_memory:
            return self.notebooks[i]
        else:
            return self.notebook_paths[i]


class APINotebookReader():
    """docstring for APINotebookReader"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.lines = []

        cells = nbformat.read(data_path, as_version=4)["cells"]
        code_cells = [c for c in cells if c["cell_type"] == 'code']
        for cell in code_cells:
            # pdb.set_trace()
            self.lines += cell["source"].split('\n')
        self.lines = [
            l for l in self.lines if l and not l.strip().startswith('#')]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]

class KaggleDiffsDataset(Dataset):

    def __init__(self, diffs, tokenizer, max_length=512,
        predict_spans=False, inserted_start_token = "<INSERTED>",
        inserted_end_token = "</INSERTED>", removed_start_token = "<REMOVED>",
        removed_end_token = "</REMOVED>", p_mask_unchanged = 0.0, replace_inserted_tokens_in_output=True):

        self.tokenizer = tokenizer
        self.diffs = diffs
        self.max_length = max_length

        self.inserted_start_token = inserted_start_token
        self.inserted_end_token = inserted_end_token
        self.removed_start_token = removed_start_token
        self.removed_end_token = removed_end_token
        self.replace_inserted_tokens_in_output = replace_inserted_tokens_in_output

        self.predict_spans = predict_spans

        self.p_mask_unchanged = p_mask_unchanged

        if self.predict_spans:
            self.span_tokens = [self.inserted_start_token,
                                self.inserted_end_token,
                                self.removed_start_token,
                                self.removed_end_token]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.span_tokens})
            self.span_token_ids = dict(
                zip(self.span_tokens, self.tokenizer.convert_tokens_to_ids(self.span_tokens)))

    def __len__(self):
        return len(self.diffs)

    def mask_unchanged_token_ids(self,token_ids, remove_span_tokens=True):
        # Assumes every start token is followed by an end token
        new_token_ids = []
        new_token_mask = []

        in_unchanged = True
        for token_id in token_ids:
            if token_id in self.span_token_ids.values():
                in_unchanged = not in_unchanged
                if remove_span_tokens:
                    continue
                else:
                    new_token_mask.append(1)
            elif in_unchanged:
                new_token_mask.append(0)
            else:
                new_token_mask.append(1)
            new_token_ids.append(token_id)

        return new_token_ids, new_token_mask

    def __getitem__(self, item):
        diff = self.diffs[item]
        orig = diff["orig"]
        new = diff["new"]

        if self.predict_spans:
            orig, new = self.compute_word_diff(orig, new, separate=True)

        orig_input = self.tokenizer(
            orig, max_length=self.max_length, truncation=True)['input_ids']
        new_output = self.tokenizer(
            new, max_length=self.max_length, truncation=True)['input_ids']

        if self.predict_spans:
            # orig_input = [x for x in orig_input if not x in self.span_token_ids.values()]
            new_output, loss_mask = self.mask_unchanged_token_ids(new_output, remove_span_tokens=self.replace_inserted_tokens_in_output)
            assert len(new_output) == len(loss_mask)

            orig_input, input_labels = self.mask_unchanged_token_ids(
                orig_input)
            if "input_labels" in diff:
                input_labels = diff["input_labels"]
        else:
            loss_mask = [1]*len(new_output)
            input_labels =[-1]*len(new_output) #Shouldn't be used unless --predict_spans

        mask_tokens = [self.tokenizer.pad_token_id,-100]
        min_length = min(len(orig_input), len(new_output))

        # randomly mask unchanged tokens to encourage variety
        for i in range(1, min_length - 1):

            if orig_input[i] == self.tokenizer.pad_token_id and new_output[i] == self.tokenizer.pad_token_id:
                break
            if orig_input[i] == new_output[i]:
                if random.uniform(0, 1) < self.p_mask_unchanged:
                    new_output[i] = self.tokenizer.pad_token_id

        # mask_tokens = [self.tokenizer.pad_token_id,-100]

        mask_tokens = [self.tokenizer.pad_token_id, -100]
        attention_mask = [int(not x in mask_tokens) for x in orig_input]
        return {"input_ids": orig_input,
                "decoder_input_ids": new_output,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "input_labels": input_labels}

    def compute_word_diff(self, a, b, semantic_cleanup=True, separate=False):
        dmp = dmp_with_pprint(inserted_start_token=self.inserted_start_token,
                              inserted_end_token=self.inserted_end_token,
                              removed_start_token=self.removed_start_token,
                              removed_end_token=self.removed_end_token)

        diff = dmp.diff_main(a, b)
        if semantic_cleanup:
            dmp.diff_cleanupSemantic(diff)

        if separate:
            return dmp.pprint_separate(diff)
        else:
            return dmp.pprint(diff)


class CoralKaggleDiffsDataset(KaggleDiffsDataset):
    """docstring for CoralKaggleDiffsDataset"""

    def __init__(self, diffs, tokenizer, max_length=512, idx2word=None, word2idx=None):
        super().__init__(diffs, tokenizer, max_length=max_length)
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        'TODO: add <pad>, <mask>, <eos>'

    def __getitem__(self, item):
        # raise NotImplementedError
        diff = self.diffs[item]
        orig = diff["orig"]
        new = diff["new"]
        orig_tree = CodeTree(orig)
        new_tree = CodeTree(new)
        orig_nodes = [self.bos_token] + \
            [tk.string for tk in orig_tree.tokens][:self.max_length - 2] + \
            [self.eos_token]
        new_nodes = [self.bos_token] + \
            [tk.string for tk in new_tree.tokens][:self.max_length - 2] + \
            [self.eos_token]
        # print(''.join(orig_nodes))
        orig_input = [self.word2idx[n] for n in orig_nodes]
        new_output = [self.word2idx[n] for n in new_nodes]
        min_length = min(len(orig_input), len(new_output))

        # randomly mask unchanged tokens to encourage variety
        for i in range(1, min_length - 1):
            if orig_input[i] == self.word2idx[self.pad_token] and new_output[i] == self.word2idx[self.pad_token]:
                break
            if orig_input[i] == new_output[i]:
                if random.uniform(0, 1) < 0.5:
                    new_output[i] = self.word2idx[self.pad_token]

        # mask_tokens = [self.tokenizer.pad_token_id,-100]

        mask_tokens = [self.word2idx[self.pad_token], -100]
        attention_mask = [int(not x in mask_tokens) for x in orig_input]
        'TODO: pass input adj mat'
        adj_mat = torch.zeros((len(orig_input), len(orig_input)))
        for k in diff["orig_tree"].ast_edges:
            if k + 1 >= len(orig_input):
                continue
            children = diff["orig_tree"].ast_edges[k]

            for c in children:
                if c + 1 >= len(orig_input):
                    continue
                adj_mat[k + 1][c + 1] = 1
                adj_mat[c + 1][k + 1] = 1
                # pdb.set_trace()
        adj_mat[0, :] = 1
        adj_mat[:, 0] = 1
        return {"input_ids": orig_input,
                # "labels": new_output,
                "decoder_input_ids": new_output,
                "attention_mask": attention_mask,
                "adj_mat": adj_mat,
                "item": [item]}


class KaggleDiffsDatasetClassification(KaggleDiffsDataset):

    # TODO: Add context parameter?

    def __init__(self, diffs, tokenizer, max_length, predict_spans=False):
        super().__init__(diffs, tokenizer, max_length=max_length)
        self.lines = []
        self.labels = []
        # pdb.set_trace()
        for diff in tqdm(self.diffs):
            removed_lines = [l for l in diff["removed"] if l.strip()]
            remained_lines = [l for l in diff["remained"] if l.strip()]

            self.lines += removed_lines

            self.labels += [[1]] * len(removed_lines)

            self.lines += remained_lines

            self.labels += [[0]] * len(remained_lines)
        items = sorted(zip(self.lines, self.labels), key=lambda x: len(x[0]))
        item_chunks = [items[i:i + 128] for i in range(0, len(items), 128)]
        random.shuffle(item_chunks)
        items = list(itertools.chain.from_iterable(item_chunks))
        self.lines, self.labels = zip(*items)

        # pdb.set_trace()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = remove_special_tokens(self.lines[item], self.tokenizer)
        input_ids = self.tokenizer(line, max_length=self.max_length, truncation=True)[
            "input_ids"]
        # need attention_mask here?
        mask_tokens = [self.tokenizer.pad_token_id, -100]
        attention_mask = [int(not x in mask_tokens) for x in input_ids]

        return {
            "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "labels": self.labels[item]
        }

class UCSDNotebookDataset(Dataset):

    def __init__(self, notebook_paths=None, notebooks=None):
        if notebook_paths is None and notebooks is None:
            raise ValueError("""Must provide either a list of notebooks
                                or a list of notebook paths""")


class APINotebookDataset(Dataset):
    """docstring for APINotebookDataset"""

    def __init__(self, lines, tokenizer, max_length):

        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = remove_special_tokens(self.lines[item], self.tokenizer)
        return {
            # ,
            "input_ids": self.tokenizer(line, max_length=self.max_length, truncation=True)["input_ids"]
            # "labels": self.labels[item]
        }


class APINotebookDatasetSeq2Seq(Dataset):

    def __init__(self, lines, tokenizer, max_length=512):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        orig = line

        orig_input = self.tokenizer(
            orig, max_length=self.max_length, truncation=True)['input_ids']

        mask_tokens = [self.tokenizer.pad_token_id, -100]
        attention_mask = [int(not x in mask_tokens) for x in orig_input]
        return {"input_ids": orig_input,
                # "labels": new_output,
                "decoder_input_ids": orig_input,
                "attention_mask": attention_mask}
