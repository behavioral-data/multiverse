import os
import errno
import requests
import glob
import os
import json

from tqdm import tqdm


from selenium import webdriver

from selenium.webdriver.firefox.options import Options 

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def download_file(url, out_path,file_mode = "wb"):
    response = requests.get(url)
    if response:
        out_file = open(out_path,file_mode)
        out_file.write(response.content)
        out_file.close()
    return response.status_code == requests.codes.ok

def version_path_to_components(path):
    slug_path, version_file =  os.path.split(path)
    
    version_id = version_file.split(".")[0]
    comp_path, slug_id = os.path.split(slug_path)
    comp_name = os.path.basename(comp_path)

    return {"version_id" : version_id,
            "slug_id" : slug_id,
            "comp_name" : comp_name}

class CompetitionReader(object):
    def __init__(self, path, python_only=False):
        self.path = path
        self.slug_ids = [os.path.basename(x) for x in glob.glob(os.path.join(self.path, "*"))]
        self.comp_name = os.path.basename(self.path)
        self.python_only = python_only

    def apply_to_slugs(self,fn):
        # Applies a function fn to a list of dicts of notebooks
        for slug_id in self.slug_ids:
            versions = self.load_slug_versions(slug_id)
            yield fn(versions)
            
    def load_slug_versions(self,slug_id):
        versions = []
        for path in glob.glob(os.path.join(self.path,slug_id,"*.json")):
            with open(path) as version_file:
                filename = os.path.basename(path)
                version_id = os.path.splitext(filename)[0]
                try:
                    version = json.load(version_file)
                    if not isinstance(version,dict):
                        continue
                except json.decoder.JSONDecodeError:
                    continue
                if self.python_only:
                    try:
                       if not version["metadata"]["language_info"]["name"] == "python":
                            continue
                    except KeyError:
                        continue
                version["version_id"] = version_id
                version["path"] = path
                versions.append(version)
        return versions


def write_jsonl(open_file, data, mode = "a"):
    for datum in data:
        open_file.write(json.dumps(datum))
        open_file.write("\n")

def load_jsonl(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            loaded_line = json.loads(line)
            lines.append(loaded_line)
    return lines

def is_git_line(line):
    return len(line) >0 and line[0] in ["+","-"]

def remove_git_chars(line):
    if is_git_line(line):
        return line[1:]
    else: 
        return line

class KaggleDiffsReader():
    def __init__(self,diff_path):
        self.diff_path = diff_path
        self.diffs = []

        with open(self.diff_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Diffs"):
                diff_line = json.loads(line)
                orig, new = self.split_orig_and_new(diff_line)
                
                diff = {
                    "metadata":diff_line["metadata"],
                    "orig":orig,
                    "new":new,
                    "cell_diff":diff_line["celll_diff"]
                }
                self.diffs.append(diff)

    def __len__(self):
        return len(self.diffs)
    
    def __getitem__(self,i):
        return self.diffs[i]
    
    def remove_git_chars(self,line):
        if line[0] in ["+","-"]:
            return line[1:]
        else: 
            return line

    def split_orig_and_new(self,diff):
        #TODO: Current preserves the plus or minus
        lines = diff["cell_diff"].split("\n")
        orig = [self.remove_git_chars(x) for x in lines if len(x)>0 and x[0] != "+" ]
        new = [self.remove_git_chars(x) for x in lines if len(x)>0 and x[0] != "-"]
        return "\n".join(orig), "\n".join(new)

def split_orig_and_new(diff):
    lines = diff.split("\n")
    orig = [remove_git_chars(x) for x in lines if len(x)>0 and x[0] != "+" ]
    new = [remove_git_chars(x) for x in lines if len(x)>0 and x[0] != "-"]
    return "\n".join(orig), "\n".join(new)

def get_inserted_and_removed(diff, as_string = False):
    lines = diff.split("\n")
    inserted = [remove_git_chars(x) for x in lines if len(x)>0 and x[0] == "+" ]
    removed = [remove_git_chars(x) for x in lines if len(x)>0 and x[0] == "-"]
    
    if as_string:
        return "\n".join(inserted), "\n".join(removed)
    else:
        return inserted, removed
