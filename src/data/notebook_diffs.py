import click
import operator
import glob
import os
import pickle

from io import StringIO
import tokenize
import token
import sys

import re
import json

from nbdime.diffing import diff_notebooks
from nbdime.diffing.generic import diff
from nbdime.utils import read_notebook
from nbdime.ignorables import diff_ignorables
from nbdime.prettyprint import PrettyPrintConfig, pretty_print_notebook_diff

from git.cmd import Git

from tqdm import tqdm

import multiprocessing
import subprocess

import utils
import pdb
START_DIFF_ESCAPE = "[[START_DIF]]"
IO_TOKEN = "[IO]"
DUMMY_VERSION_TOKEN = "[VERSION{}]"

HUNK_ID_REGEX = r"@@\s([+\-][0-9]*,[0-9]*\s){2}@@"
IO_REGEX = r"(open|print|read_(csv|txt|json))\s*\(.*\)"
DUMMY_VERSION_REGEX = r"[VERSION[0-9]*]"


class DummyInclude:
    def __init__(self):
        self.sources = True
        self.outputs = False
        self.attachments = False
        self.metadata = False
        self.details = False


class CollectDiffs:
    def __init__(self, ignore_empty_lines=True, ignore_line_shuffle=False,
                 print_diffs=False):

        # self.next_is_modified = False
        # self.out_filename = os.path.join(out_path,"notebook_diffs.txt")
        # self.out_file = open(self.out_filename,"w")

        self.current_diff = ""
        self.current_version_path = None
        self.next_is_modified = False
        self.competition_diffs = []
        self.print_diffs = print_diffs

        # If a modification simply adds or deletes lines,
        # or the only change is whitespace, don't save it
        self.ignore_empty_lines = ignore_empty_lines

        self.ignore_line_shuffle = ignore_line_shuffle

    def write(self, text):
        if self.is_end_of_modified(text):
            self.next_is_modified = False
            self.save_cell_diff()
        elif "## modified" in text:
            self.next_is_modified = True
        elif self.next_is_modified:
            self.current_diff = self.current_diff + self.remove_hunk_id(text)

        return text

    def clean_dummy_token_lines(self, text):
        lines = text.split("\n")

        new_lines = []
        for line in lines:
            if re.findall(DUMMY_VERSION_REGEX, line):
                line = re.sub(DUMMY_VERSION_REGEX, "", line)
                line = " " + line[1:]
            new_lines.append(line)
        return "\n".join(new_lines)

    def save_cell_diff(self):
        if self.current_diff_is_valid():

            # Not sure why this is, but sometimes the hunk ids
            # end up inline, rather than above the start of the hunk.
            diff_to_save = self.remove_special_chars(self.current_diff)

            output = {"original_path": self.current_version_path,
                      "metadata": utils.version_path_to_components(self.current_version_path),
                      "cell_diff": diff_to_save}

            if self.print_diffs:
                print(diff_to_save)
                print("-" * 10)
            self.competition_diffs.append(output)

        self.current_diff = ""

    def current_diff_is_valid(self):
        if self.ignore_empty_lines or self.ignore_line_shuffle:
            old_lines = [utils.remove_git_chars(x) for x in self.current_diff.split(
                "\n") if len(x) > 0 and x[0] == "-"]
            new_lines = [utils.remove_git_chars(x) for x in self.current_diff.split(
                "\n") if len(x) > 0 and x[0] == "+"]

        if self.ignore_empty_lines:
            non_whitespace_new = [x for x in new_lines if not x.isspace()]
            non_whitespace_old = [x for x in old_lines if not x.isspace()]
            if len(non_whitespace_new) == 0 or len(non_whitespace_old) == 0:
                return False

        if self.ignore_line_shuffle:
            if set(new_lines) == set(old_lines):
                return False

        return True

    def is_end_of_modified(self, text):
        non_mod_delim = ["## deleted", "## inserted", START_DIFF_ESCAPE]
        return self.next_is_modified and any(x in text for x in non_mod_delim)

    def is_start_of_hunk(self, text):
        return re.search(HUNK_ID_REGEX, text)

    def remove_special_chars(self, text):
        text = self.remove_hunk_id(text)
        text = self.clean_dummy_token_lines(text)
        return text

    def remove_hunk_id(self, text):
        return re.sub(HUNK_ID_REGEX, "", text)


def source_as_string(notebook):
    for cell in notebook["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])

    return notebook


def remove_comments_from_src(src):
    """
    This reads tokens using tokenize.generate_tokens and recombines them
    using tokenize.untokenize, and skipping comment/docstring tokens in between
    """
    lines = src.split("\n")
    new_lines = [x for x in lines if len(
        x.strip()) > 0 and x.strip()[0] != "#"]
    return "\n".join(new_lines)

# Maybe we want to do this in the model's preprocessing?
# That way we can more explicitly set the probability of generating
# IO_TOKEN in a suggestion.


def encode_io_in_src(src):
    return re.sub(IO_REGEX, IO_TOKEN, src)

# Can't get output diff working, not sure it's important anyway


def remove_outputs(notebook):
    for cell in notebook["cells"]:
        cell["outputs"] = []
    return notebook


def remove_large_cells(notebook, threshold=10000):
    new_cells = []
    for cell in notebook["cells"]:
        if not cell.get("source"):
            continue
        if len(cell["source"]) < threshold:
            new_cells.append(cell)
    notebook["cells"] = new_cells
    return notebook


def remove_markdown(notebook):
    new_cells = []
    for cell in notebook["cells"]:
        if not cell["cell_type"] == "markdown":
            new_cells.append(cell)
    notebook["cells"] = new_cells
    return notebook


def remove_comments(notebook):
    new_cells = []
    for i in range(len(notebook["cells"])):
        cell = notebook["cells"][i]
        if cell["cell_type"] == "code":
            cell["source"] = remove_comments_from_src(cell["source"])
        new_cells.append(cell)
    notebook["cells"] = new_cells
    return notebook


def encode_io_in_notebook(notebook):
    new_cells = []
    for i in range(len(notebook["cells"])):
        cell = notebook["cells"][i]
        if cell["cell_type"] == "code":
            cell["source"] = encode_io_in_src(cell["source"])
        new_cells.append(cell)
    notebook["cells"] = new_cells
    return notebook


def load_kaggle_notebook_for_nbdime(path, ignore_comments=False, encode_io=False):
    notebook = read_notebook(path, on_null="empty")
    notebook = remove_outputs(notebook)
    notebook = remove_large_cells(notebook)
    notebook = remove_markdown(notebook)
    if ignore_comments:
        notebook = remove_comments(notebook)
    if encode_io:
        notebook = encode_io_in_notebook(notebook)
    return notebook


def add_dummy_version_tokens(notebook, index):
    """In order to get the differ to output full versions of the notebook,
       we add dummy tokens to the cell such that nbdiff will return a
       modified operation for those cells which it would otherwise ignore"""
    new_cells = []
    for i in range(len(notebook["cells"])):
        cell = notebook["cells"][i]
        if cell["cell_type"] == "code":
            cell["source"] = cell["source"] + "\n" + \
                DUMMY_VERSION_TOKEN.format(index)
        new_cells.append(cell)
    notebook["cells"] = new_cells
    return notebook

def get_diff_sequential_submissions(print_config, ignore_comments=False, encode_io=False,
                                    inverted_mode=False):

    def diff_sequential_submissions(versions):
        versions = sorted(versions, key=lambda x: x["version_id"])
        def version_loader(x): return load_kaggle_notebook_for_nbdime(x["path"], ignore_comments=ignore_comments,
                                                                      encode_io=encode_io)
        version_notebooks = list(map(version_loader, versions))

        for i, (a, b) in enumerate(zip(version_notebooks, version_notebooks[1:])):
            if inverted_mode:
                if i == 0:
                    a = add_dummy_version_tokens(a, i)
                b = add_dummy_version_tokens(b, i + 1)

            diffs = diff_notebooks(a, b)
            print_config.out.current_version_path = versions[i]["path"]
            pretty_print_notebook_diff(
                START_DIFF_ESCAPE, "b", a, diffs, print_config)

    return diff_sequential_submissions


def set_minimal():
    subprocess.Popen("git config --global diff.algorithm minimal")


def unset_minimal():
    subprocess.Popen("git config --global diff.algorithm myers")


# Has to be on the top level so that the
# pickler in imap can access it
DIFF_WORKER_PARAMS = {"ignore_comments": False,
                      "use_color": False,
                      "python_only": False,
                      "ignore_empty_lines": False,
                      "ignore_line_shuffle": False,
                      "encode_io": False,
                      "print_diffs": False,
                      "inverted_mode": False}


def diff_worker(comp_path):
    pdb.set_trace()
    python_only = DIFF_WORKER_PARAMS["python_only"]
    encode_io = DIFF_WORKER_PARAMS["encode_io"]
    competition = utils.CompetitionReader(comp_path, python_only=python_only)

    ignore_empty_lines = DIFF_WORKER_PARAMS["ignore_empty_lines"]
    ignore_line_shuffle = DIFF_WORKER_PARAMS["ignore_line_shuffle"]

    print_diffs = DIFF_WORKER_PARAMS["print_diffs"]
    competition_diff_collector = CollectDiffs(ignore_empty_lines=ignore_empty_lines,
                                              ignore_line_shuffle=ignore_line_shuffle,
                                              print_diffs=print_diffs)

    use_color = DIFF_WORKER_PARAMS["use_color"]
    ignore_comments = DIFF_WORKER_PARAMS["ignore_comments"]

    inverted_mode = DIFF_WORKER_PARAMS["inverted_mode"]

    print_config = PrettyPrintConfig(include=DummyInclude(), out=competition_diff_collector,
                                     use_git=True, use_color=use_color)

    list(competition.apply_to_slugs(get_diff_sequential_submissions(print_config,
                                                                    ignore_comments=ignore_comments,
                                                                    encode_io=encode_io,
                                                                    inverted_mode=inverted_mode)))
    return competition_diff_collector.competition_diffs


@click.command()
@click.argument('competitions_path', type=click.Path())
@click.argument('out_path', type=click.Path())
@click.option("--git_context", type=int, default=1000, help="Lines of context for git diffs")
@click.option("--ignore_comments", is_flag=True, default=False)
@click.option("--use_color", is_flag=True, default=False)
@click.option("--python_only", is_flag=True, default=False)
@click.option("--ignore_empty_lines", is_flag=True, default=False)
@click.option("--ignore_line_shuffle", is_flag=True, default=False)
@click.option("--encode_io", is_flag=True, default=False)
@click.option("--n_workers", type=int, default=4)
@click.option("--print_diffs", is_flag=True, default=False)
@click.option("--inverted_mode", is_flag=True, default=False, help="Return lines that don't change and are\
                                 outside the diff context")
def main(competitions_path, out_path, git_context, ignore_comments, use_color, python_only,
         ignore_empty_lines, ignore_line_shuffle, encode_io, n_workers, print_diffs, inverted_mode):
    comp_paths = glob.glob(os.path.join(
        competitions_path, "*", ""))  # pylint: disable=undefined-variable

    DIFF_WORKER_PARAMS["ignore_comments"] = ignore_comments
    DIFF_WORKER_PARAMS["use_color"] = use_color
    DIFF_WORKER_PARAMS["python_only"] = python_only
    DIFF_WORKER_PARAMS["ignore_empty_lines"] = ignore_empty_lines
    DIFF_WORKER_PARAMS["ignore_line_shuffle"] = ignore_line_shuffle
    DIFF_WORKER_PARAMS["encode_io"] = encode_io
    DIFF_WORKER_PARAMS["print_diffs"] = print_diffs

    DIFF_WORKER_PARAMS["inverted_mode"] = inverted_mode

    if inverted_mode:
        git_context = 1000

    _environ = dict(os.environ)
    try:
        os.environ["GIT_DIFF_OPTS"] = "-u{}".format(git_context)

        if n_workers == 1:
            yield_results = map(diff_worker, comp_paths)
        else:
            p = multiprocessing.Pool(n_workers)
            yield_results = p.imap_unordered(diff_worker, comp_paths)

        with open(os.path.join(out_path, "every_cell.jsonl"), "w") as out_file:

            p = multiprocessing.Pool(n_workers)
            with tqdm(total=len(comp_paths)) as pbar:
                for result in yield_results:
                    if not print_diffs:
                        for line in result:
                            out_file.write(json.dumps(line))
                            out_file.write("\n")
                    pbar.update()
    finally:

        os.environ.clear()
        os.environ.update(_environ)


if __name__ == "__main__":
    main()
