import glob
import os 
import pandas as pd
import json
import ast
from tqdm import tqdm
import click
import pickle
from multiprocessing import Pool, cpu_count, Queue

from functools import partial

import itertools


import sys
sys.setrecursionlimit(15000)

import logging

logpath = "./tree_matches.log"
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
ch = logging.FileHandler(logpath)
# ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)


def replace_function_subtrees(coral_repr):
    ignore = []
    new_tree = []
    for i in range(len(coral_repr)):
        node = coral_repr[i]
        if i in ignore:
            #ignore the children too:
            ignore = ignore + node.get("children",[])
            continue
        elif node["type"] == "Call":
            ignore = ignore + node.get("children",[])[1:]
        new_tree.append(node)
    return new_tree
            
class Snippet(object):
    def __init__(self,slug,version_id,source,competition = None,
                max_size=512):

        self.slug = slug
        self.max_size = max_size
        self.version_id = version_id
        self.source = source
        self.coral_repr = parse_string(source)[:self.max_size]
        self.function_args_removed_repr = replace_function_subtrees(self.coral_repr)
        self.python_ast = ast.parse(source)
    
    def coral_diff(self,other,key = None,attr="coral_repr"):
        a_attr = getattr(self,attr)
        b_attr = getattr(other,attr)
        return self.tree_diff(a_attr, b_attr, key = key)
        
    def rear_pad_list(a,n):
        m = len(a)
        return a + [None for i in range(n-m)]
    
    def make_same_length(a,b):
        n = max(len(a),len(b))
    
        a = rear_pad_list(a,n)
        b = rear_pad_list(b,n)
    
        return (a,b)
    
    def tree_diff(self,a,b,key=None):
        a,b = make_same_length(a,b)
        if not key:
            key =  lambda aa,bb: not aa == bb
        return sum([key(aa,bb) for (aa,bb) in zip(a,b)])
    
    def to_dict(self):
        return {"slug":self.slug, "version_id" : self.version_id, 
               "source":self.source}


def rear_pad_list(a,n):
    m = len(a)
    return a + [None for i in range(n-m)]
    
def make_same_length(a,b):
    n = max(len(a),len(b))
    
    a = rear_pad_list(a,n)
    b = rear_pad_list(b,n)
    
    return (a,b)
    
def tree_diff(a,b,key=None):
    a,b = make_same_length(a,b)
    if not key:
        key =  lambda aa,bb: not aa == bb
    return sum([key(aa,bb) for (aa,bb) in zip(a,b)])

def looks_like_string(node):
    node_type = node.get("type")
    if node_type == "Constant":
        try:
            float(node.get("value"))
            return False
        except (ValueError,TypeError):
            return True
    else:
        return False
    
def dont_count_strings(a,b):
    if a is None or b is None:
        return True
    if looks_like_string(a) and looks_like_string(b):
        return False
    else:
        return (not a == b)

def remove_duplicate_matches(matches):
    to_return = []
    record = set()
    for match in matches:
        if not (match[0].source,match[1].source) in record:
            record.add((match[0].source,match[1].source))
            to_return.append(match)
    return to_return

# def get_matching_cells(kernel_trees,diff_versions = False, key = None):
#     matches = []
#     all_cells = []
#     for slug,versions in kernel_trees.items():
#         all_version_cells = []
#         for version_id, cells in versions.items():
#             if cells:
#                 for cell in cells:
#                     all_version_cells.append(cell)

#         n = len(all_version_cells)
#         if n == 1:
#             continue
#         for i in range(n):
#             for j in range(i+1,n):
                    
#                 cell_i = all_version_cells[i]
#                 cell_j = all_version_cells[j]
                
#                 if diff_versions:
#                     if cell_i.version_id == cell_j.version_id:
#                         continue
#                 diff = cell_i.coral_diff(cell_j,key=key)
#                 if diff == 1: 
#                     matches.append((cell_i,cell_j))
#         all_cells = all_cells + all_version_cells
#     return matches
def sort_versions_by_version_id(dictionary):
    tuples = list(dictionary.items())
    return sorted(tuples, key=lambda x : int(x[0]))

def get_sequential_matching( kernel_trees, key=None, attr="coral_repr"):
    matches = []
    for slug,versions in kernel_trees.items():
        sorted_versions = sort_versions_by_version_id(versions)
        for a,b in zip(sorted_versions, sorted_versions[1:]):
            a_version_id, a_cells = a
            b_version_id, b_cells = b
            for a_cell in a_cells:
                for b_cell in b_cells:
                    diff = a_cell.coral_diff(b_cell, key=key, attr=attr)
                    if diff == 1:
                        matches.append((a_cell, b_cell))
    return matches

def get_matching_cells(kernel_trees,diff_versions = False, key = None,attr="coral_repr"):
    matches = []
    all_cells = []
    for slug,versions in kernel_trees.items():
        all_version_cells = []
        for version_id, cells in versions.items():
            if cells:
                for cell in cells:
                    all_version_cells.append(cell)

        n = len(all_version_cells)
        if n == 1:
            continue
        for i in range(n):
            for j in range(i+1,n):
                    
                cell_i = all_version_cells[i]
                cell_j = all_version_cells[j]
                
                if diff_versions:
                    if cell_i.version_id == cell_j.version_id:
                        continue
                diff = cell_i.coral_diff(cell_j,key=key,attr=attr)
                if diff == 1:
                    matches.append((cell_i,cell_j))
        all_cells = all_cells + all_version_cells
    return matches

def parse_string(string):
    global c, d
    tree = ast.parse(string)
    
    json_tree = []
    def gen_identifier(identifier, node_type = 'identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos
    
    def traverse_list(l, node_type = 'list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item))
        if (len(children) != 0):
            json_node['children'] = children
        return pos
        
    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = str(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s
        elif isinstance(node, ast.alias):
            json_node['value'] = str(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = str(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.keyword):
            json_node['value'] = str(node.arg)
        

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.With):
            children.append(traverse(node.context_expr))
            if node.optional_vars:
                children.append(traverse(node.optional_vars))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.Try):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.handlers, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args'))
            children.append(traverse_list(node.defaults, 'defaults'))
            if node.vararg:
                children.append(gen_identifier(node.vararg, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type'))
            if node.name:
                children.append(traverse_list([node.name], 'name'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases'))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child, ast.boolop) or isinstance(child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child))
                
        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))
                
        if (len(children) != 0):
            json_node['children'] = children
        return pos
    
    traverse(tree)
    return json_tree

def get_param_from_filename(param,filename):
    template = "\?{}=(.*)\.|\?"
    query_regex = re.compile(template.format(param))
    try:
        return re.findall(query_regex,filename)[0]
    except IndexError:
        return None
    
def get_slug_from_file(filename):
    return re.split("\?|\.",filename)[0]

def load_cell_as_snippets(slug,version_id,path,max_size=512):
    with open(path) as kernel_file:
        cells = []

        try:
            res = json.load(kernel_file)
        except ValueError:
            return cells
            
        if not (type(res) is dict) or not "cells" in res:
            return cells
            
        for cell in res["cells"]:
            if not cell.get("source"):
                continue
            if type(cell["source"]) is list:
               cell["source"] = "".join(cell["source"])
            
            try:
                cells.append(Snippet(slug,version_id,cell["source"],max_size=max_size))
                                                    
            except (SyntaxError, AttributeError):
                continue
        return cells
            

def get_slug_matches(competition_path,slug,ignore_function_args=False,
                    remove_exact_duplicates=False,
                    length_threshold=None, ignore_strings=False,max_size=512,
                    sequential_matches=False):
    # in_path is a slug directory
    
    kernel_version_snippets = {slug:{}}
    for version_path in glob.glob(os.path.join(competition_path,slug,"*.json")):

        filename = os.path.basename(version_path)
        version_id = os.path.splitext(filename)[0]
        
        if not version_id:
            continue
    
        version_snippets = load_cell_as_snippets(slug,version_id,version_path,max_size=max_size)
        kernel_version_snippets[slug][version_id] = version_snippets
    
    if ignore_function_args:
        match_attr = "function_args_removed_repr"
    else:
        match_attr = "coral_repr"

    if ignore_strings:
        key = dont_count_strings
    else:
        key = None

    if sequential_matches:
        matches = get_sequential_matching(kernel_version_snippets,key=key, attr=match_attr)
    else:
        matches = get_matching_cells(kernel_version_snippets, diff_versions = True,
                                key=key, attr=match_attr)
    
   
    if length_threshold:
        matches=[x for x in matches if len(x[0].source.split("\n")) > 5]
    
    if remove_exact_duplicates:
        matches = remove_duplicate_matches(matches)
    return matches
        

# def get_competition_matches(competition_path):
    
#     slugs = [os.path.basename(x) for x in glob.glob(os.path.join(competition_path,"*"))]
#     matches = []
#     for slug in slugs:
#         matches = matches + get_slug_matches(competition_path,slug)
#     logger.info("Done with {}".format(competition_path))
#     return matches

def get_competition_matches(ignore_function_args,length_threshold,remove_exact_duplicates,
                            ignore_strings, max_size, sequential_matches, competition_path):
        
        slugs = [os.path.basename(x) for x in glob.glob(os.path.join(competition_path,"*"))]
        matches = []
        for slug in tqdm(slugs):
            matches = matches + get_slug_matches(competition_path,slug,ignore_function_args,
                                                remove_exact_duplicates, length_threshold, ignore_strings,
                                                max_size,sequential_matches)
        logger.info("Done with {}".format(competition_path))

        return matches  
        
# def get_competition_matcher(ignore_function_args,length_threshold,remove_exact_duplicates,
#                             ignore_strings):
#     def get_competition_matches(ignore_function_args,length_threshold,remove_exact_duplicates,
#                             ignore_strings, competition_path):
        
#         slugs = [os.path.basename(x) for x in glob.glob(os.path.join(competition_path,"*"))]
#         matches = []
#         for slug in slugs:
#             matches = matches + get_slug_matches(competition_path,slug,ignore_function_args,
#                                                 remove_exact_duplicates, length_threshold, ignore_strings)
#         logger.info("Done with {}".format(competition_path))

#         return matches  

#     return get_competition_matches

def write_matches(out_path,matches):
    with open(os.path.join(out_path,"matches.jsonl"), 'w') as the_file:
        for match in matches:
            the_file.write(json.dumps([match[0].to_dict(),match[1].to_dict()]))
            the_file.write("\n")

@click.command()
@click.argument('in_path', type=click.Path())
@click.argument('out_path', type = click.Path())
@click.option('--ignore_function_args', is_flag = True, default=False, show_default=True)
@click.option('--length_threshold', default=None, show_default=True)
@click.option('--remove_exact_duplicates',is_flag = True, default=False, show_default=True)
@click.option('--ignore_strings', is_flag = True,default=False, show_default=True)
@click.option('--max_size', default=512, show_default=True)
@click.option('--sequential_matches', is_flag=True, default=False,show_default=True)
def main(in_path,
        out_path,
        ignore_function_args,
        length_threshold,
        remove_exact_duplicates,
        ignore_strings,
        max_size,
        sequential_matches):
    all_comp_paths = glob.glob(os.path.join(in_path,"*"))[1:2]
    n = len(all_comp_paths)
    # all_matches = map(get_competition_matches,all_comp_paths)
    all_matches = []
    
    comp_matcher = partial(get_competition_matches,ignore_function_args,
                                        length_threshold,
                                        remove_exact_duplicates,
                                        ignore_strings,
                                        max_size,
                                        sequential_matches)
    all_matches = [comp_matcher(all_comp_paths[0])]
    # with Pool(16) as pool:
    #     for result in tqdm(pool.imap_unordered(comp_matcher,all_comp_paths),total =n):
    #         all_matches.append(result)
    #     pool.join()
    #     pool.close()

    # with Pool(8) as worker_pool: 
    #     all_matches = tqdm(worker_pool.imap_unordered(get_competition_matches,all_comp_paths),total =n)
    
    all_matches = itertools.chain.from_iterable(all_matches)
    write_matches(out_path,all_matches)
        

if __name__ == '__main__':
    main()
    