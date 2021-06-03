import click
import glob
import os
import nbformat
from nbformat.reader import NotJSONError

import json
from nbconvert.exporters import PythonExporter
import multiprocessing
from tqdm import tqdm

from tokenizers.processors import BertProcessing


"""
Loads a directory of notebooks and writes scripts to a different directory
"""



_PYTHON_ONLY = True
_OUT_DIR = None 
CELL_SEP_TOKEN = "<CELL_SEP>"
_ONE_FILE = False

def write_script(data, old_path):
    new_name = os.path.basename(old_path).split(".")[0] + ".py"
    new_path = os.path.join(_OUT_DIR, new_name)
    with open(new_path, 'w') as f:
        f.write(data)

def process_notebook(path_to_notebook):
    try:
        notebook = nbformat.read(path_to_notebook, as_version=4)
        if _PYTHON_ONLY and notebook["metadata"]["language_info"]["name"] != "python":
            return
        output = CELL_SEP_TOKEN.join([cell["source"] for cell in notebook["cells"] if cell["cell_type"]=="code"])
    
    except (AttributeError, KeyError, NotJSONError, TypeError, UnboundLocalError):
        return

    if _ONE_FILE:
        return output
    else:
        write_script(output,path_to_notebook)

@click.command()
@click.argument("in_dir", type=click.Path())
@click.argument("out_dir", type=click.Path())
@click.option("--n_workers", type=click.INT, default=32)
@click.option("--python_only", is_flag=True, default=True)
@click.option("--one_big_file", is_flag=True, default=False)
def main(in_dir, out_dir, n_workers, python_only, one_big_file):
    global _PYTHON_ONLY
    _PYTHON_ONLY = python_only
    
    global _OUT_DIR
    _OUT_DIR = out_dir

    global _ONE_FILE
    _ONE_FILE = one_big_file

    in_paths = glob.glob(os.path.join(in_dir,"*.ipynb"))

    if n_workers == 1:
        yield_results = map(process_notebook,in_paths)
    else:
        p = multiprocessing.Pool(n_workers)  
        yield_results = p.imap_unordered(process_notebook, in_paths)
    
    if one_big_file:
        with open(os.path.join(out_dir,"all_python_cells_sep.txt"), "w") as out_file :
            with tqdm(total=len(in_paths)) as pbar:
                for result in yield_results:
                    if result:
                        out_file.write(json.dumps([result]))
                        out_file.write("\n")
                        pbar.update()

    
    else:
        with tqdm(total=len(in_paths)) as pbar:
            for result in yield_results:
                pbar.update()




if __name__ == "__main__":
    main()