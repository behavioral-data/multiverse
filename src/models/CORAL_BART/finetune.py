import json
import os
import logging
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError
from tqdm import tqdm
import random
from transformers import (BartForConditionalGeneration, DataCollator,
                          RobertaTokenizerFast, Trainer, TrainingArguments,
                          BartTokenizerFast, BartConfig, BartForSequenceClassification,
                          AutoTokenizer, BertForTokenClassification, BertConfig)

from src.models.CORAL_BART.trainer import CORALBARTTrainer, CORALBARTTrainerClassification, CORALBARTTrainerSeq2Seq, CORALBARTMultiTaskTrainer
from src.models.CORAL_BART.dataset import (KaggleDiffsDataset, KaggleDiffsDatasetClassification, KaggleDiffsReader,
                                           DynamicPaddingCollator, DynamicPaddingCollatorSeq2Seq, CoralKaggleDiffsDataset, CoralDiffsReader)
from src.models.CORAL_BART.metrics import get_seq2seq_eval, classification_eval, get_multitask_eval
from src.models.CORAL_BART.utils import count_parameters, block_shuffle, has_internet
from src.models.CORAL_BART.models import MultiTaskBart, HyperbolicLibraryEmbedding

from src.data.scrape_library_structures import TokenizedGraph

import numpy as np
import click


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)



@click.command()
@click.argument("path_to_dataset", type=click.Path())
@click.option("--cuda_device", default="0", help="CUDA id")
@click.option("--model_name", default=None, help="Path to a pytorch config.json or a HuggingFace s3 bucket. If none, start from scratch")
@click.option("--encoder_attention_heads", default=16)
@click.option("--decoder_attention_heads", default=16)
@click.option("--encoder_layers", default=12)
@click.option("--decoder_layers", default=12)
@click.option("--path_to_tokenizer", default='./tokenizer/')
@click.option("--max_length", default=512)
@click.option("--emb_size", type=int, default=None)
@click.option("--max_size", default=-1, help="maximum number of diffs to read - only for debug")
@click.option("--num_train_epochs", default=3)
@click.option("--per_device_train_batch_size", default=4)
@click.option("--per_device_eval_batch_size", default=4)
@click.option("--warmup_steps", default=50)
@click.option("--save_steps", default=10000, help="Steps between model cache")
@click.option("--weight_decay", default=0.01)
@click.option("--learning_rate", default=5e-5, type=float)
@click.option("--logging_dir", default="./logs/")
@click.option("--logging_steps", default=500)
@click.option("--eval_steps", default=1000, help="Steps between model evaluations")
@click.option("--save_eval", is_flag=True, default=False)
@click.option("--eval_fraction", default=0.1, help="Fraction of the dataset to reserve for evaluation")
@click.option("--task", default="seq2seq", help="Only option (for now) is seq2seq")
@click.option("--output_dir", default="./results/")
@click.option("--task", default="seq2seq")
@click.option("--wandb", is_flag=True, default=False)
@click.option("--face", is_flag=True, default=False)
@click.option("--predict_spans", is_flag=True, default=False, help="Use masked loss to focus on changed tokens")
@click.option("--span_aware_decoding", is_flag=True, default=False, help="Train the model for span-aware decoding (include span tokens in decoder output)")
@click.option("--span_aware_loss", is_flag=True, default=False, help="Only calculate loss over spans that change")
@click.option("--pos_class_weight", type=float, default=1.0, help="Weight for the positive class in the classification loss weight")
@click.option("--unchanged_loss_weight", type=float, default=None, help="If a value is provided, weight the loss over unchanged spans by this factor (factor over decision points is 1)")
@click.option("--coral", is_flag=True, default=False)
@click.option("--p_mask_unchanged", default=0.0, help="Probability with which to mask unchanged tokens (don't calculate loss over them)")
@click.option("--classification_threshold", default=0.1, help="If the softmax output for the positive label for an input_id is greater than this value, label it as positive")
@click.option("--initial_tree_embeddings", type=click.Path(), help="Path to a directory which contains a tree.pickle of libraries and a model checkpoint from pytorch_hyperbolic.py")
@click.option("--library_graph", type=click.Path(), help="Path to a tree.pickle of libraries. If provided, also minimize the loss from Ge et. al 2019 (ICLR)")
@click.option("--graph_loss_weight", type=float, default=1.0)
@click.option("--classification_loss_weight", type=float, default=1.0)
@click.option("--graph_loss_burn_in_epochs", type=float, default=0.0, help="Number of epochs until the graph loss kicks in")
@click.option("--notes", type=str, default=None, help="Notes to save to wandb")
@click.option("--tags", multiple=True, default=None, help="List of tags to save to wandb")
@click.option("--dont_evaluate", is_flag=True, default=False)
@click.option("--oracle_span_aware_decoder", is_flag=True, default=False)
@click.option("--oracle_mixin_p",type=float, default=1.0, help="Probability with which binary token predicions are flipped to equal the Oracle")
@click.option("--forced_acc",type=float, default=None, help="Force the model's predictions to be this F1 be comparing to ground truth")
@click.option("--deterministically_shuffle_dataset", is_flag=True, default=False)
@click.option("--hidden_dropout_prob",type=float, default=0.0)
@click.option("--train_fraction",type=float, default=1.0, help="Fraction of training set to use")
@click.option("--randomize_training",is_flag=True, help="Shuffle the training order (breaks up folds)")
@click.option("--no_train",is_flag=True, help="Don't train the model")
@click.option("--no_eval",is_flag=True, help="Don't evalute the model")
def main(path_to_dataset,
         cuda_device=0,
         model_name=None,
         encoder_attention_heads=16,
         decoder_attention_heads=16,
         encoder_layers=12,
         decoder_layers=12,
         path_to_tokenizer='./tokenizer/',
         max_length=512,
         max_size=-1,
         emb_size=None,
         output_dir='./results/',
         task="seq2seq",
         num_train_epochs=3,
         per_device_train_batch_size=4,
         per_device_eval_batch_size=16,
         warmup_steps=500,
         learning_rate=5e-5,
         save_steps=10000,
         weight_decay=0.01,
         logging_dir="./logs/",
         logging_steps=500,
         eval_steps=10000,
         save_eval=True,
         evaluate_during_training=True,
         eval_fraction=0.1,
         wandb=False,
         face=False,
         predict_spans=False,
         coral=False,
         p_mask_unchanged=0.,
         initial_tree_embeddings=None,
         library_graph=None,
         graph_loss_weight=1.0,
         classification_loss_weight = 1.0,
         notes=None,
         tags=None,
         span_aware_decoding=False,
         span_aware_loss=True,
         unchanged_loss_weight=None,
         pos_class_weight=1.0,
         classification_threshold=0.1,
         dont_evaluate=False,
         oracle_span_aware_decoder=False,
         oracle_mixin_p=1.0,
         forced_acc=None,
         deterministically_shuffle_dataset=False,
         graph_loss_burn_in_epochs=0,
         hidden_dropout_prob=0.0,
         train_fraction=1.0, 
         randomize_training=False,
         no_train=False,
         no_eval=False):

    if wandb:
        import wandb
        run = wandb.init(project="robustdatascience", job_type=task, entity="mikeamerrill",
                         notes=notes, tags=tags)
        if has_internet():
            run.save()
            output_dir = os.path.join(output_dir, run.name)
        else:
            logger.info(
                "Can't connect to wandb. Maybe running on cluster node?")
            output_dir = os.path.join(output_dir, run.id)

    # Validate arguments:
    if task == "multi" and not predict_spans:
        raise AttributeError(
            "--predict_spans needs to be enabled to perform multitask task")

    if (span_aware_decoding or span_aware_loss) and not predict_spans:
        raise AttributeError(
            "--predict_spans needs to be enabled to use span aware decoding or span aware loss")
    if emb_size is None:
        emb_size = max_length
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)

    vocab_path = os.path.join(path_to_tokenizer, "vocab.json")
    merges_path = os.path.join(path_to_tokenizer, "merges.txt")
    tokenizer = BartTokenizerFast(vocab_path, merges_path)

    if task == "seq2seq":
        if span_aware_decoding:
            raise NotImplementedError(
                "span-aware decoding needs a language model head AND a token-level classification head")
        base_model = BartForConditionalGeneration
        if coral:
            base_dataset = CoralKaggleDiffsDataset
        else:
            base_dataset = KaggleDiffsDataset

        base_trainer = CORALBARTTrainerSeq2Seq
        dataset_args = {"predict_spans": predict_spans,
                        "p_mask_unchanged": p_mask_unchanged,
                        "replace_inserted_tokens_in_output": False}

        base_trainer = CORALBARTTrainerSeq2Seq
        config = BartConfig(vocab_size=tokenizer.vocab_size,
                            d_model=max_length,
                            encoder_attention_heads=encoder_attention_heads,
                            decoder_attention_heads=decoder_attention_heads,
                            encoder_layers=encoder_layers,
                            decoder_layers=decoder_layers)
        collator = DynamicPaddingCollatorSeq2Seq(tokenizer)
        compute_metrics = get_seq2seq_eval(tokenizer)

    elif task == "classification":
        base_model = BertForTokenClassification
        base_dataset = KaggleDiffsDataset
        dataset_args = {"predict_spans": predict_spans}

        base_trainer = CORALBARTTrainerClassification
        config = BertConfig(vocab_size=tokenizer.vocab_size,
                            d_model=max_length,
                            encoder_attention_heads=encoder_attention_heads,
                            decoder_attention_heads=decoder_attention_heads,
                            encoder_layers=encoder_layers,
                            decoder_layers=decoder_layers,
                            num_labels=2,
                            )
        collator = DynamicPaddingCollatorSeq2Seq(tokenizer)
        compute_metrics = classification_eval

    elif task == "multi":
        base_model = MultiTaskBart
        # if coral:
        #     raise NotImplementedError("Multitask CORAl Encoder not yet")
        base_dataset = KaggleDiffsDataset
        dataset_args = {"predict_spans": predict_spans,
                        "p_mask_unchanged": p_mask_unchanged,
                        "replace_inserted_tokens_in_output": False}

        base_trainer = CORALBARTMultiTaskTrainer
        config = BartConfig(vocab_size=tokenizer.vocab_size,
                            d_model=max_length,
                            encoder_attention_heads=encoder_attention_heads,
                            decoder_attention_heads=decoder_attention_heads,
                            encoder_layers=encoder_layers,
                            decoder_layers=decoder_layers,
                            num_labels=2,
                            hidden_dropout_prob=hidden_dropout_prob,
                            span_aware_decoding=span_aware_decoding,
                            classification_threshold=classification_threshold)

        collator = DynamicPaddingCollatorSeq2Seq(tokenizer)
        compute_metrics = get_multitask_eval(
            tokenizer, wandb=wandb, threshold=classification_threshold)

    else:
        raise NotImplementedError(
            "task must be one of {seq2seq, classification, multi}")

    if model_name:
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        config.vocab_size = len(tokenizer) + 4
        model = base_model.from_pretrained(model_name, config=config)
        config = model.config

    else:
        logger.info("Training from scratch...")
        model = base_model(config)

    if initial_tree_embeddings:
        init_emebeddings = HyperbolicLibraryEmbedding(
            initial_tree_embeddings, tokenizer)
        logger.info("Initializing model with tree emebeddings...")
        init_emebeddings.set_embeddings(model)

    model.cuda()

    logger.info("Model Config: {}".format(model.config.to_diff_dict()))
    n_params = count_parameters(model)
    logger.info("Number of trainable parameters: {}".format(n_params))

    if coral:
        data_reader = CoralDiffsReader(path_to_dataset, max_size=max_size)
    else:
        data_reader = KaggleDiffsReader(path_to_dataset, max_size=max_size)
    
    if deterministically_shuffle_dataset:
        #Shuffle deterministically
        data_reader.diffs = block_shuffle(
            data_reader.diffs, key_fn=lambda x: x["metadata"]["comp_name"],
            seed=421994)
    else:
        data_reader.diffs = block_shuffle(
            data_reader.diffs, key_fn=lambda x: x["metadata"]["comp_name"])
    n_examples = len(data_reader)
    split_point = int(n_examples * (1 - eval_fraction))

    eval_diffs = data_reader.diffs[split_point:]
    train_diffs = data_reader.diffs[:split_point]
    
    if randomize_training:
        random.shuffle(train_diffs)
    
    train_diffs = train_diffs[:int(split_point*train_fraction)]
    
    if coral:
        train_dataset = base_dataset(train_fraction,tokenizer, 
                    max_length=max_length, idx2word=data_reader.idx2word, 
                    word2idx=data_reader.word2idx)
        eval_set = base_dataset(eval_diffs,tokenizer, 
                    max_length=max_length, idx2word=data_reader.idx2word, 
                    word2idx=data_reader.word2idx)

        with open(os.path.join(output_dir, 'vocab.json'), 'w') as fout:
            fout.write(json.dumps({"idx2word": data_reader.idx2word,
                                   "word2idx": data_reader.word2idx}))
    else:
        train_dataset = base_dataset(
            train_diffs, tokenizer, max_length=max_length, **dataset_args)
        eval_set = base_dataset(
            eval_diffs, tokenizer, max_length=max_length, **dataset_args)

    # Necessary because we add new tokens for the span prediction task
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        logging_first_step=True,
        learning_rate=learning_rate,
        save_total_limit=2,
        evaluate_during_training=not dont_evaluate
    )

    trainer = base_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_set,
        prediction_loss_only=False,
        compute_metrics=compute_metrics,
        data_collator=collator,
        tokenizer=tokenizer,
        save_eval=True,
        # face=face,
        tb_writer=None,
        library_graph=library_graph,
        graph_loss_burn_in_epochs=graph_loss_burn_in_epochs,
        graph_loss_weight=graph_loss_weight,
        classification_loss_weight = classification_loss_weight,
        span_aware_loss=span_aware_loss,
        unchanged_loss_weight=unchanged_loss_weight,
        oracle_span_aware_decoder=oracle_span_aware_decoder,
        oracle_mixin_p=oracle_mixin_p,
        forced_acc=forced_acc,
        pos_class_weight=pos_class_weight
    )

    if not no_train:
        trainer.train()
    if not no_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
