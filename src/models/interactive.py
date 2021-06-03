from transformers import BartForConditionalGeneration, RobertaTokenizerFast, Trainer, TrainingArguments, DataCollator
import os
from glob import glob

def load_model(path):
    model = BartForConditionalGeneration.from_pretrained(path)
    return model

def load_model_wandb(wandb_id, results_path = "results/"):
    model_path = os.path.join(results_path,wandb_id,"*","")
    model_checkpoints = glob(model_path)
    most_recent_checkpoint = sorted(model_checkpoints)[-1]
    return load_model(most_recent_checkpoint)

def get_alternative_interface(model, tokenizer):
    def interface(text):
        inputs = tokenizer(text, return_tensors="pt")
        # outputs = model.generate(inputs["input_ids"], max_length=250, do_sample=True, top_p=0.95, top_k=60)
        outputs = model.generate(
                        inputs["input_ids"], 
                        max_length=50, 
                        num_beams=5, 
                        early_stopping=True)

        return ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs.tolist()])[0]
    return interface

def process_input(model,tokenizer,inputs):
    outputs = model.generate(
                        inputs["input_ids"], 
                        max_length=50, 
                        num_beams=5, 
                        early_stopping=True)

    return ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs.tolist()])[0]
