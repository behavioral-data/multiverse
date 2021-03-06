MULTIVERSE
==============================

In response to an exponential increase in available data and a growing culture of “data-driven discovery”,
analysis tools have enjoyed widespread innovation and adoption. However, while data systems proliferate,
robust processes to guide the use of these systems remain in relatively short supply.

This project serves to address these challenges by developing ML-powered tools for encouraging best practices. 

How to contibute:
------------
1. Clone this repo: `git clone https://github.com/behavioral-data/RobustDataScience.git`
2. cd into it:  `cd RobustDataScience`
3. Build the conda environment: `make create_environment`
4. Follow the project structure outlined below.

Getting data
------------
You can also ust `make` to create the data used in this project:
1. Get a Kaggle api key by creating an account and going to User Icon > My Account > api
2. Add `KAGGLE_USERNAME=<username> and  KAGGLE_KEY=<key>` to `.env`.
3. Run `make data` 

Why is this project set up like this?
------------
Great question. For a more satisfying answer than can be provided here, look to the [original cookiecutter page](https://drivendata.github.io/cookiecutter-data-science/): 

One thing that I really like about using the setup is that it's really easy to modularize code.
Say that you wrote some really handy function in `src/visualization.py` that you wanted to use in a notebook. One option might have been to write your notebook in the main directory, and use `src` as a module. This is all well and good for notebook, but what if you have several? A more heinous (and common!) idea might have been to copy and paste over the code to your notebook. 
However, since we turned `src` into a module and added it to the PATH in `make create_environment`, we can just do something like this in our notebook, no matter where it is in the project:
```
from src.visualization import handy_viz
handy_viz(df)
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── RobustDataScience.yml <- yml for building the conda environment
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── train_model.py
    │   │   └── CORAL-LM       <- The CORAL Model from the KDD2020 submission 
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
