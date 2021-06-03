from transformers import RobertaTokenizerFast
import click
import glob
import os

@click.command()
@click.argument("scripts_path", type=click.Path())
@click.argument("tokenizer_save_path", type=click.Path())
def main(scripts_path,tokenizer_save_path):
    paths = glob.glob(os.path.join(scripts_path,"*.py"))
    tokenizer = RobertaTokenizerFast()
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    print(tokenizer_save_path)
    tokenizer.save_model(tokenizer_save_path, "")

if __name__ == "__main__":
    main()