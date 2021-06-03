import click
import os
import numpy as np
from transformers import (BartForConditionalGeneration, DataCollator,
                          RobertaTokenizerFast, Trainer, TrainingArguments,
                          BartTokenizerFast, BartConfig, BartForSequenceClassification,
                          AutoTokenizer, BertForTokenClassification, BertConfig)
from src.models.CORAL_BART.trainer import CORALBARTTrainer, CORALBARTTrainerClassification, CORALBARTTrainerSeq2Seq, CORALBARTMultiTaskTrainer
from src.models.CORAL_BART.dataset import (KaggleDiffsDataset, KaggleDiffsDatasetClassification, KaggleDiffsReader,
                                           DynamicPaddingCollator, DynamicPaddingCollatorSeq2Seq, CoralKaggleDiffsDataset, CoralDiffsReader)
from src.models.CORAL_BART.utils import count_parameters, block_shuffle, has_internet
from src.models.CORAL_BART.metrics import get_seq2seq_eval, classification_eval, get_multitask_eval
from src.models.CORAL_BART.models import MultiTaskBart
from scipy.special import softmax

@click.command()
@click.argument("path_to_dataset", type=click.Path())
@click.argument("path_to_class_model", type=click.Path())
@click.argument("path_to_gen_model", type=click.Path())
@click.option("--path_to_tokenizer", type=click.Path(), default='./tokenizer/')
def main(path_to_dataset,
         path_to_class_model,
         path_to_gen_model,
         path_to_tokenizer='./tokenizer/'):
    
    vocab_path = os.path.join(path_to_tokenizer, "vocab.json")
    merges_path = os.path.join(path_to_tokenizer, "merges.txt")
    tokenizer = BartTokenizerFast(vocab_path, merges_path)
    collator = DynamicPaddingCollatorSeq2Seq(tokenizer)

    base_dataset = KaggleDiffsDataset
    dataset_args = {"predict_spans": True,
                    "replace_inserted_tokens_in_output": False}
    data_reader = KaggleDiffsReader(path_to_dataset)
    data_reader.diffs = block_shuffle(
            data_reader.diffs, key_fn=lambda x: x["metadata"]["comp_name"],
            seed=421994)
    
    n_examples = len(data_reader)
    split_point = int(n_examples * (1 - 0.05))    
    eval_dataset = base_dataset(
          data_reader.diffs[split_point:], tokenizer, max_length=128, **dataset_args)

    class_training_args = TrainingArguments(
        output_dir="./results/",
        overwrite_output_dir=False,
        # num_train_epochs=num_train_epochs,
        # per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=150,
        # warmup_steps=warmup_steps,
        # save_steps=save_steps,
        # weight_decay=weight_decay,
        # logging_dir=logging_dir,
        # logging_steps=logging_steps,
        # eval_steps=eval_steps,
        # logging_first_step=True,
        # learning_rate=learning_rate,
        # save_total_limit=2,
        # evaluate_during_training=not dont_evaluate
    )
    class_model = BertForTokenClassification.from_pretrained(path_to_class_model)
    class_trainer =  CORALBARTTrainerClassification( model=class_model,
                    args=class_training_args,
                    eval_dataset=eval_dataset,
                    prediction_loss_only=False,
                    compute_metrics=classification_eval,
                    data_collator=collator,
                    tokenizer=tokenizer,
                    save_eval=True)
    eval_data_loader = class_trainer.get_eval_dataloader()
    output = class_trainer._prediction_loop(eval_data_loader , description="Evaluation")
    
    for i,(logits,labels) in enumerate(zip(output.predictions, output.label_ids)):
        labels = np.array(labels)
        logits = np.array(logits)
        mask = labels != -100

        labels= labels[mask]
        logits = logits[mask]
        score = softmax(logits,axis=1)[:,-1]
        pred_labels = score > 0.15
        eval_dataset.diffs[i]["input_labels"] = pred_labels.tolist()
    
    gen_metrics = get_multitask_eval(
            tokenizer, wandb=False, threshold=0.15)
    gen_config = BartConfig(
        vocab_size = tokenizer.vocab_size+4,
        hidden_dropout_prob=0.0,
        num_labels=2,
        span_aware_decoding=True,
        classification_threshold=0.15,
        d_model=128
    )
    gen_model =  MultiTaskBart.from_pretrained(path_to_gen_model, config=gen_config)
    gen_training_args = TrainingArguments(
        output_dir="./results/",
        overwrite_output_dir=False,
        # num_train_epochs=num_train_epochs,
        # per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=60,
        # warmup_steps=warmup_steps,
        # save_steps=save_steps,
        # weight_decay=weight_decay,
        # logging_dir=logging_dir,
        # logging_steps=logging_steps,
        # eval_steps=eval_steps,
        # logging_first_step=True,
        # learning_rate=learning_rate,
        # save_total_limit=2,
        # evaluate_during_training=not dont_evaluate
    )
    
    gen_trainer = CORALBARTMultiTaskTrainer(model=gen_model,
                    args=gen_training_args,
                    eval_dataset=eval_dataset,
                    prediction_loss_only=False,
                    data_collator=collator,
                    tokenizer=tokenizer,
                    save_eval=True,
                    compute_metrics = gen_metrics,
                    oracle_span_aware_decoder=True)
    gen_trainer.evaluate()

if __name__ == "__main__":

    main()