from flask import Flask, flash, request, redirect, url_for, render_template
import os
from bs4 import BeautifulSoup, Tag
import pdb
import nbformat
from nbconvert import HTMLExporter
import json


from transformers import (DataCollator, RobertaTokenizerFast, Trainer, BartForConditionalGeneration,
                          BartTokenizerFast, BartConfig, BartForSequenceClassification, TrainingArguments, AutoTokenizer)

from src.models import KaggleDiffsReader, APINotebookDataset, APINotebookDatasetSeq2Seq, DynamicPaddingCollatorSeq2Seq, DynamicPaddingCollator, CORALBARTTrainerSeq2Seq, CORALBARTTrainerClassification, safe_decode, MultiTaskBart, KaggleDiffsDataset, CORALBARTMultiTaskTrainer
# ============ variables for model =============
path_to_tokenizer = '/projects/bdata/tmp/tokenizer/'
model_name_seq2seq = '/projects/bdata/tmp/rural-spaceship-48'
model_name_classification = '/projects/bdata/jupyter/gezhang_backup/RobustDataScience/src/models/CORAL_BART/coral_seq2seq/checkpoint-20000'
model_name = '/projects/bdata/tmp/autumn-salad-68/checkpoint-160000'
encoder_attention_heads = 8
decoder_attention_heads = 8
encoder_layers = 12
decoder_layers = 12
max_length = 512

# ==============================================

os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

# load tokenizer
print('Loading tokenizer')
vocab_path = os.path.join(path_to_tokenizer, "vocab.json")
merges_path = os.path.join(path_to_tokenizer, "merges.txt")
tokenizer = BartTokenizerFast(vocab_path, merges_path)

# load model
print('Loading model')
# base_model_seq2seq = BartForConditionalGeneration
# base_model_classification = BartForSequenceClassification
# model_seq2seq = base_model_seq2seq.from_pretrained(model_name_seq2seq)
# model_classification = base_model_classification.from_pretrained(
#     model_name_classification)

config = BartConfig(vocab_size=tokenizer.vocab_size + 4,
                    d_model=max_length,
                    encoder_attention_heads=encoder_attention_heads,
                    decoder_attention_heads=decoder_attention_heads,
                    encoder_layers=encoder_layers,
                    decoder_layers=decoder_layers,
                    num_labels=2,
                    hidden_dropout_prob=0.0)

base_model = MultiTaskBart
model = base_model(config)
model = model.from_pretrained(model_name, config=config)

html_exporter = HTMLExporter()
app = Flask(__name__)

# Path to Kaggle notebooks
location = os.path.abspath(os.path.dirname(__file__))
notebook_path = '/projects/bdata/jupyter/_7_1'
# notebook_path = os.path.join(location, './notebooks')
# path = os.path.join(location, './notebooks')


def get_cells_from_notebook(content):
    pdb.set_trace()
    raise NotImplementedError


@app.route('/')
def index():
    return 'Hello, World!'


@app.route('/coral/<filename>', methods=["POST", "GET"])
def coral(filename):
    if filename.endswith('.ipynb'):
        if not os.path.exists(os.path.join(location, './templates', filename.replace('.ipynb', '.html'))):
            with open(os.path.join(notebook_path, filename), 'r') as f:
                content = f.read()
            t = nbformat.reads(content, as_version=4)
            (content, resources) = html_exporter.from_notebook_node(t)
            content = '  <script src="annotate.js" type="text/javascript">  </script>\n  <link href="annotate.css" rel="stylesheet" type="text/css">\n'.format(filename.split('.')[0]) \
                + content

            soup = BeautifulSoup(content)
            cells = soup.find_all("div", class_="input_area")
            cells_content = []
            lines = []
            for cell in cells:
                text = cell.find("pre").text
                c_lines = text.split('\n')
                lines += c_lines
                cells_content.append(text)
            dataset = KaggleDiffsDataset(
                [{"orig": l, "new": l} for l in lines], tokenizer, max_length=max_length)
            collator = DynamicPaddingCollatorSeq2Seq(tokenizer)

            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)
            training_args = TrainingArguments(
                output_dir='./',
                overwrite_output_dir=False,
                per_device_eval_batch_size=20,
            )
            base_trainer = CORALBARTMultiTaskTrainer

            trainer = base_trainer(
                  model=model,
                  args=training_args,
                  train_dataset=dataset,
                  eval_dataset=dataset,
                  prediction_loss_only=False,
                  data_collator=collator,
                  tokenizer=tokenizer,
                  save_eval=True  )

            results = trainer.demo_predict(dataset)
            # output_seq2seq = trainer_seq2seq.predict(
            #     dataset_seq2seq).predictions
            # output_seq2seq = [safe_decode(tokenizer, p)
            #                   for p in output_seq2seq]
            # output_classification = trainer_classification.predict(
            #     dataset_classification).predictions
            # output_classification = output_classification.argmax(axis=1)
            # print(output_classification)
            # print(output_seq2seq)
            i = 0
            for cell in cells:
                # print(cell)
                c_lines = cell.find("pre").text.split('\n')
                span_lines = cell.find("pre").decode_contents().split('\n')


                for ii, span_line in enumerate(span_lines):
                    # if 1 in results["classification"][i]:
                    # if output_classification[i] == 1:
                        # span_lines[ii] = span_line.replace(
                        #     '<span', '<span style="background-color: yellow"')
                    span_lines[ii] = span_line.replace(
                            '<span', '<span alternatives="{}"'.format(tokenizer.decode( results["generation"][i] if 2 not in results["generation"][i] else results["generation"][i][:results["generation"][i].index(2)])))
                    i += 1
                new_pre = BeautifulSoup('\n'.join(span_lines))
                # pdb.set_trace()
                for child_orig, child_new in zip(cell.find("pre").children, new_pre.body.children):
                    # pdb.set_trace()
                    if isinstance(child_new, str):
                        continue
                    if "style" in child_new.attrs or True:
                        child_orig["class"] = "dp"
                        # child_orig.attrs["alternatives"] = 'A,B,C'
                        child_orig.attrs["alternatives"] = child_new.attrs["alternatives"]
                        new_node = soup.new_tag("ul")
                        new_node["class"] = "dropdown-content"
                        input_node = soup.new_tag("input")
                        input_node["type"] = "text"
                        input_node["id"] = "myInput"
                        new_node.append(input_node)
                        add = soup.new_tag("span")
                        add["onclick"] = "newElement(this)"
                        add["class"] = "addBtn"
                        add.string = "Add"
                        new_node.append(add)
                        for j in child_orig.attrs["alternatives"].split('#&&&#'):
                            son = soup.new_tag("li")
                            son.string = j
                            grand_son = soup.new_tag("span")
                            grand_son.string = "x"
                            grand_son["class"] = "close"
                            grand_son["onclick"] = "delete_alt(this)"
                            son.append(grand_son)
                            new_node.append(son)
                        child_orig.append(new_node)

            # pdb.set_trace()
            # cell.find("pre").string.replace_with('\n'.join(span_lines))
            with open(os.path.join(location, './templates', filename.replace('.ipynb', '.html')), 'w') as fout:
                fout.write(soup.prettify())

        return render_template(filename.replace('.ipynb', '.html'))

    else:
        return render_template(filename.replace('.ipynb', '.html'))

    raise NotImplementedError


if __name__ == '__main__':
    host = '0.0.0.0'
    app.secret_key = 'super secret key'

    debug = True
    app.run(host=host, debug=debug)
