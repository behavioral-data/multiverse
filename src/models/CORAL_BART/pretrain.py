import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import random
import glob
import pdb
import itertools
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import json
import asttokens
import ast

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    # AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    # Trainer,
    TrainingArguments,
    set_seed,
    BartTokenizerFast,
    # BartForConditionalGeneration,
    BartConfig
)
from debug_trainer import Trainer
from modeling_auto import AutoModelForSeq2SeqLM
from modeling_bart import BartForConditionalGeneration
from dataset import TypeDataCollatorForLanguageModeling

from tqdm import tqdm

logger = logging.getLogger(__name__)

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
CELL_SEP = "<CELL_SEP>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "The maximum input sequence length"}
    )
    encoder_attention_heads: Optional[int] = field(
        default=16, metadata={"help": "number of encoder attention heads"}
    )
    decoder_attention_heads: Optional[int] = field(
        default=16, metadata={"help": "number of decoder attention heads"})
    encoder_layers: Optional[int] = field(
        default=16, metadata={"help": "number of encoder layers"})
    decoder_layers: Optional[int] = field(
        default=16, metadata={"help": "number of decoder layers"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The input directory for training data."}
    )

    big_file_path: Optional[str] = field(
        default=None, metadata={"help": "Big file with all cells in it"}
    )
    big_file_path_kaggle: Optional[str] = field(default=None, metadata={"help": "Big file of Kaggle with all cells in it"})
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={
            "help": "Overwrite the cached training and evaluation sets"}
    )
    eval_frac: float = field(
        default=0.1,
        metadata={
            "help": "Fraction of dataset reserved for evaluation"
        },
    )


def get_dataset_from_paths(paths, tokenizer: PreTrainedTokenizer, evaluate=False, max_length=512):
    return ScriptsDataset(tokenizer, paths, max_length)


def get_tree_from_ast_node(node):
    '''
    params:
        node: ast.__Node__
    return:
        Node(node)
    '''
    root = Node({"type": type(node), "startpos": node.first_token.startpos,
                 "endpos": node.last_token.endpos})
    for n in ast.iter_child_nodes(node):
        if hasattr(n, 'first_token'):
            root.children.append(get_tree_from_ast_node(n))
    return root


class Node():
    def __init__(self, data):
        '''
        data: {'type': ast.__NodeType__,
               'startpos': int,
               'endpos': int}
        '''
        self.type = data["type"]
        self.startpos = data["startpos"]
        self.endpos = data["endpos"]

        self.children = []

    def __repr__(self):
        return "<Node type: {} startpos: {} endpos: {} children: {}>".format(self.type, self.startpos, self.endpos, self.children)

    def insert(self, data):
        sp = data["startpos"]
        ep = data["endpos"]
        is_son = True
        for child in self.children:
            if sp >= child.startpos and ep <= child.endpos:
                child.insert(data)
                is_son = False
                break
        if is_son:
            self.children.append(Node(data))

    def search_span_type(self, startpos, endpos):
        for n in self.children:
            if startpos >= n.startpos and endpos <= n.endpos:
                return n.search_span_type(startpos, endpos)
        return self.type


class ScriptsDataset(Dataset):
    def __init__(self, tokenizer, paths, evaluate: bool = False, max_length=512, chunk_size=128):
        paths = paths[:100000]
        self.examples = []
        for p in tqdm(paths, desc="Tokenizing"):
            lines = [x for x in Path(p).read_text(
                encoding="utf-8").split(CELL_SEP) if len(x) > 0]
            if len(lines) == 0:
                continue
            self.examples += tokenizer(lines, max_length=max_length,
                                       truncation=True)["input_ids"]
        sorted_examples = sorted(self.examples, key=lambda x: len(x))
        chunked_examples = [sorted_examples[i:i + chunk_size]
                            for i in range(0, len(sorted_examples), chunk_size)]
        random.shuffle(chunked_examples)
        self.examples = list(itertools.chain.from_iterable(chunked_examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


class ScriptsDatasetFromSnippets(ScriptsDataset):
    def __init__(self, tokenizer, snippets, evaluate: bool = False, max_length=512):

        self.examples = tokenizer(snippets, max_length=max_length, truncation=True)[
            "input_ids"]
        self.max_length = max_length


class KaggleDatasetFromSnippets(Dataset):
    def __init__(self, tokenizer, snippets, evaluate: bool = False, max_length=512, path_to_type_vocab='/projects/bdata/jupyter/gezhang_backup/RobustDataScience/dataset/type_vocab.json'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.type_dict = {'idx2type': [], 'type2idx': {}}
        with open(path_to_type_vocab, 'r') as f:
            self.type_dict["idx2type"] = json.load(f)
        self.type_dict["type2idx"] = {
            t: i for i, t in enumerate(self.type_dict["idx2type"])}
        self.type_dict["type2idx"]['pad'] = -1
        # pdb.set_trace()
        self.snippets = [l for l in snippets if len(l) <= 1600]
        self.examples = [None for l in self.snippets]
        return
        for line in tqdm(snippets, desc="Tokenizing"):
            if len(line) > 1600:
                continue
            try:
                results = tokenizer(line, max_length=max_length,
                                    truncation=True, return_offsets_mapping=True)
                # if 1 in results["input_ids"]:
                #     pdb.set_trace()
                # if len(results["input_ids"]) > 1000:
                # pdb.set_trace()
                # continue
                atok = asttokens.ASTTokens(line, parse=True)
                root = atok.tree
                new_tree = get_tree_from_ast_node(root)
                types = [new_tree.search_span_type(
                    span[0], span[1]) for span in results["offset_mapping"]]
                results["type"] = [self.type_dict["type2idx"].get(t.__name__, self.type_dict["type2idx"]['pad'])
                                   for t in types]
                self.examples.append(results)
            except Exception as e:
                pass
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        if self.examples[i] is not None:
            return {'input_ids': torch.tensor(self.examples[i]["input_ids"]),
                    'type': torch.tensor(self.examples[i]["type"])}

        else:

            try:
                line = self.snippets[i]
                results = self.tokenizer(line, max_length=self.max_length,
                                         truncation=True, return_offsets_mapping=True)
                atok = asttokens.ASTTokens(line, parse=True)
                root = atok.tree
                new_tree = get_tree_from_ast_node(root)
                types = [new_tree.search_span_type(
                    span[0], span[1]) for span in results["offset_mapping"]]
                results["type"] = [self.type_dict["type2idx"].get(t.__name__, self.type_dict["type2idx"]['pad'])
                                   for t in types]
                self.examples[i] = results
                # self.examples.append(results)
            except Exception as e:
                print(e)
                line = ""
                results = self.tokenizer(line, max_length=self.max_length,
                                         truncation=True, return_offsets_mapping=True)
                atok = asttokens.ASTTokens(line, parse=True)
                root = atok.tree
                new_tree = get_tree_from_ast_node(root)
                types = [new_tree.search_span_type(
                    span[0], span[1]) for span in results["offset_mapping"]]
                results["type"] = [self.type_dict["type2idx"].get(t.__name__, self.type_dict["type2idx"]['pad'])
                                   for t in types]
                self.examples[i] = results
                # self.examples.append(results)
                # pass
            return {'input_ids': torch.tensor(self.examples[i]["input_ids"]),
                    'type': torch.tensor(self.examples[i]["type"])}
        # return {'input_ids': torch.tensor(self.examples[i]["input_ids"]),
        #         'type': torch.tensor(self.examples[i]["type"])}
        # return torch.tensor(self.examples[i]["input_ids"])


def load_jsonl(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            loaded_line = json.loads(line)
            lines.append(loaded_line)
    return lines


def load_examples_from_path(path):
    notebooks = load_jsonl(path)
    random.shuffle(notebooks)
    to_return = []
    for notebook in [x[0] for x in notebooks]:
        to_return += [x for x in notebook.split(CELL_SEP) if len(x) > 0]
    return to_return


def load_exmaples_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    train_examples = data["train"]
    valid_examples = data["valid"]
    return train_examples, valid_examples
    # pdb.set_trace()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    'please check if weight_decay is set'
    # pdb.set_trace()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    vocab_file = os.path.join(model_args.tokenizer_name, "vocab.json")
    merges_file = os.path.join(model_args.tokenizer_name, "merges.txt")
    tokenizer = BartTokenizerFast(vocab_file, merges_file)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = BartConfig(vocab_size=tokenizer.vocab_size,
                            d_model=model_args.max_length,
                            encoder_attention_heads=model_args.encoder_attention_heads,
                            decoder_attention_heads=model_args.decoder_attention_heads,
                            encoder_layers=model_args.encoder_layers,
                            decoder_layers=model_args.decoder_layers,
                            n_types=56)
        # pdb.set_trace()
        print('================={model_args.max_length}')
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # pdb.set_trace()
    if model_args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = BartForConditionalGeneration(config)
        logger.info('!' * 20)

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = model_args.max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, model_args.max_length)

    # Get datasets
    if data_args.data_dir:
        all_paths = glob.glob(os.path.join(data_args.data_dir, "*.py"))
        random.shuffle(all_paths)
        n_paths = len(all_paths)
        train_eval_split = int((1 - data_args.eval_frac) * n_paths)
        train_paths = all_paths[:train_eval_split]
        eval_paths = all_paths[train_eval_split:]

        train_dataset = get_dataset_from_paths(
            train_paths, tokenizer=tokenizer, max_length=model_args.max_length) if training_args.do_train else None
        eval_dataset = get_dataset_from_paths(
            eval_paths, tokenizer=tokenizer, max_length=model_args.max_length) if training_args.do_eval else None

    elif data_args.big_file_path:
        all_examples = load_examples_from_path(data_args.big_file_path)
        n_examples = len(all_examples)
        train_eval_split = int((1 - data_args.eval_frac) * n_examples)

        train_examples = all_examples[:train_eval_split]
        eval_examples = all_examples[train_eval_split:]

        train_dataset = ScriptsDatasetFromSnippets(
            tokenizer, train_examples, max_length=model_args.max_length)
        eval_dataset = ScriptsDatasetFromSnippets(
            tokenizer, eval_examples, max_length=model_args.max_length)

    elif data_args.big_file_path_kaggle:
        train_examples, eval_examples = load_exmaples_from_json(
            data_args.big_file_path_kaggle)
        'data set, which should assign token type'
        train_dataset = KaggleDatasetFromSnippets(
            tokenizer, train_examples[:int(1 * len(train_examples))], max_length=model_args.max_length)
        eval_dataset = KaggleDatasetFromSnippets(
            tokenizer, eval_examples[:int(1 * len(eval_examples))], max_length=model_args.max_length)

        # pdb.set_trace()

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    # )

    data_collator = TypeDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=data_collator,
    )

    # Initialize our Trainer
    # pdb.set_trace()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # pdb.set_trace()
        trainer.train(model_path=model_path)
        # pdb.set_trace()
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # pdb.set_trace()

        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    if training_args.do_predict:
        pred_dataset = KaggleDatasetFromSnippets(
            tokenizer, eval_examples[:int(0.001 * len(eval_examples))], max_length=model_args.max_length)
        pred_results = trainer.predict(pred_dataset)
        print(pred_results)
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
