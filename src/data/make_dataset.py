# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from kaggle_tools import fetch_kernels_for_competition
from kaggle_tools import main as kaggle_main
import os



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making data set')
    
    logger.info("""Using kaggle API to download all competition submissions and all versions of those submissions """)
    competition_dir = os.path.join(output_filepath, "competitions")
    Path(competition_dir).mkdir(parents=True, exist_ok=True)


    # fetch_kernels_for_competition("ieee-fraud-detection", competition_dir,
    #                             save_kernel_metadata = True, progress_bar = True,
    #                             fetch_versions = True)
    
    kaggle_main(competition_dir)
    logger.info("make data complete")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
