import click
import operator
import glob
import os
import pickle


from nbdime.diffing import diff_notebooks
from nbdime.diffing.generic import diff
from nbdime.utils import read_notebook

from tqdm import tqdm
# from nbdime.diffing.notebooks import compare_cell_approximate, compare_cell_moderate, compare_cell_strict, compare_output_approximate, notebook_differs


import utils

def clean_slug(out_path):
    pass
@click.command()
@click.argument('competitions_path', type=click.Path())
@click.argument('out_path', type=click.Path())
def main(competitions_path, out_path):
    comp_paths = glob.glob(os.path.join(competitions_path, "*", ""))[1:2] # pylint: disable=undefined-variable
    for comp_path in tqdm(comp_paths):
        competition = utils.CompetitionReader(comp_path)
        n = len(competition.slug_ids)
        results = list(competition.apply_to_slugs(diff_sequential_submissions),total=n)
        

if __name__ == "__main__":
    main()