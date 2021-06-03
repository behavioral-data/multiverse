import random
import pprint
pp = pprint.PrettyPrinter(indent=4)

from src.models.CORAL_BART.dataset import KaggleDiffsReader, KaggleDiffsDataset
import click
from rouge import Rouge

@click.group()
def cli():
    pass

@cli.command()
@click.argument("path_to_dataset", type=click.Path())
@click.option("--max_size", type=int, default=1000)
def identity(path_to_dataset, max_size):
    diffs = [x for x in KaggleDiffsReader(path_to_dataset)]
    random.shuffle(diffs)
    diffs = diffs[:max_size]
    prediction_strings = [x["orig"] for x in diffs]
    label_strings = [x["new"] for x in diffs]
    
    scores = {}
    rouge = Rouge()

    rogue_scores = rouge.get_scores(
            prediction_strings, label_strings, avg=True, ignore_empty=True)

    scores["rouge-l-p"] = rogue_scores["rouge-l"]["p"]
    scores["rouge-l-f"] = rogue_scores["rouge-l"]["f"]
    scores["rouge-l-r"] = rogue_scores["rouge-l"]["r"]
    pp.pprint(scores)

if __name__ == "__main__":
    cli()