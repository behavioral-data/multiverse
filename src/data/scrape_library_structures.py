
import os
import json
from enum import Enum
import inspect
import pkgutil
import sys
import re
import click
import pickle
import sys
sys.setrecursionlimit(10000)

# from treelib import Tree, Node
# from treelib.exceptions import DuplicatedNodeIdError

import networkx as nx
from tqdm import tqdm
import importlib
import pandas as pd

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import dijkstra

from transformers import BartTokenizerFast

class PackageNode():
    MODULE = 1
    CLASS = 2
    FUNCTION = 3
    PARAM = 4


class TokenizedGraph():
    def __init__(self, G, tokenizer, use_cache=True):
        self.tokenizer = tokenizer

        self.n     = G.order()
        self.cache = dict() if use_cache else None

        self.tokenizer = tokenizer   
        self.tokenize_graph(G)
        # self.input_id_to_d_matrix_index = {i:self.input_id_graph_indices.index(self.input_id_to_graph_matrix_id[i]) for i in self.supported_input_ids}

    def tokenize_graph(self,G): 
        G_copy = G.copy()
        self.supported_input_ids=set()

        #Add the token nodes
        for node_id, data in G.nodes(data=True):
            node_token_ids = self.tokenizer.encode(data["name"],add_special_tokens=False)
            for nt_id in node_token_ids:
                self.supported_input_ids.add(nt_id)
                G_copy.add_edge(node_id,nt_id)
        
        G_copy = G_copy.to_undirected()
        G_int = nx.convert_node_labels_to_integers(G_copy,label_attribute="old_id")
        
        input_id_to_graph_matrix_id = {data["old_id"]:i for i,data in G_int.nodes(data=True) if data["old_id"] in self.supported_input_ids}
        graph_matrix_id_to_input_id = {i:data["old_id"] for i,data in G_int.nodes(data=True) if data["old_id"] in self.supported_input_ids}

        G_matrix = nx.to_scipy_sparse_matrix(G_int)
        
        input_id_graph_indices = list(input_id_to_graph_matrix_id.values())
        _distances = dijkstra(G_matrix, directed=False,unweighted=True,
                                            indices=input_id_graph_indices)
        self.input_id_distances = _distances[:len(input_id_graph_indices),input_id_graph_indices]
        # This point is annoying but important to preserve
        # the mapping where BPE adds the Ġ character (for spaces)
        self.input_id_to_d_matrix_index = {i:input_id_graph_indices.index(input_id_to_graph_matrix_id[i]) for i in self.supported_input_ids}
        for input_id, d_matrix_id in list(self.input_id_to_d_matrix_index.items()):
            token = self.tokenizer.convert_ids_to_tokens([input_id])[0]   
            space_token = " " + token
            space_token_ids = self.tokenizer.encode(space_token, add_special_tokens=False)
            if len(space_token_ids) == 1:
                space_token_id = space_token_ids[0]
                self.input_id_to_d_matrix_index[space_token_id] = d_matrix_id
                self.supported_input_ids.add(space_token_id)


    def get_pairwise_graph_distances(self,input_ids):
        n = len(input_ids)
        distances = np.empty((n,n)) 
        distances[:]  = np.nan
        for i,i_id in enumerate(input_ids):
            distances[i,i] = 0
            for j, j_id in enumerate(input_ids[:i]):
                if (not i_id in self.supported_input_ids) or  (not j_id in self.supported_input_ids):
                    continue
                else:
                    graph_distance_matrix_i = self.input_id_to_d_matrix_index[i_id]
                    graph_distance_matrix_j = self.input_id_to_d_matrix_index[j_id]
                    distances[i,j]= self.input_id_distances[graph_distance_matrix_i,graph_distance_matrix_j]
                    distances[j,i]= self.input_id_distances[graph_distance_matrix_i,graph_distance_matrix_j]
        return distances

def load_jsonl(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            loaded_line = json.loads(line)
            lines.append(loaded_line)
    return lines


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    try:
        package_path = package.__path__
    except AttributeError:
        return results

    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        
        full_name = package.__name__ + '.' + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except :
            continue
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

def add_param_nodes(fcn,parent,tree):
    try:
        params = inspect.signature(fcn).parameters.values()
    except ValueError:
        return
        
    for param in inspect.signature(fcn).parameters.values():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            param_id = ".".join([parent,param.name])
            tree.add_node(param_id, name = param.name, kind = PackageNode.PARAM)
            tree.add_edge(parent,param_id)
            # tree.create_node(param.name, param_id, parent=parent, data = PackageNode.PARAM)


def paste_at_root(G,H,G_parent):
    G.add_nodes_from(H.nodes(data=True))
    G.add_edges_from(H.edges(data=True))

    H_root = [n for n,d in H.in_degree() if d==0][0]
    
    G.add_edge(G_parent, H_root)
    return G

def join_graphs(G,H,u,v):
    G.add_nodes_from(H.nodes(data=True))
    G.add_edges_from(H.edges(data=True))
    G.add_edge(u,v)
    return G

def get_tree_for_pkg_better(package,parent_id=None,lib_name=None,max_depth=None,depth=0):

    root_name = str(package.__name__).split(".")[-1]
    if parent_id:
        root_id = parent_id + "." + root_name
    else:
        root_id = root_name 
    
    G = nx.DiGraph()
    G.add_node(root_id, name=root_name)
    
    if depth == max_depth:
        return root_id, G
    
    try:
        memberiter = inspect.getmembers(package)
    except TypeError:
        return root_id, G

    for name, obj in inspect.getmembers(package):
        if name[0] == "_" or "tests" in name or name in root_id:
            continue
        
        #Ignore anything that's not defined in this library

        elif inspect.isclass(obj) or inspect.ismodule(obj):
            try:
                if lib_name and not lib_name in inspect.getfile(obj):
                    continue
            except TypeError:
                continue
    
            sub_root_id, class_graph = get_tree_for_pkg_better(obj,parent_id=root_id,lib_name=lib_name,
                                                                max_depth=max_depth, depth=depth+1)
            G = join_graphs(G,class_graph,root_id,sub_root_id)
        
        elif inspect.isfunction(obj):
            function_id = ".".join([root_id,obj.__name__])
            G.add_node(function_id, name=obj.__name__)
            G.add_edge(root_id,function_id)
            # add_param_nodes(obj,function_id,G)
    return root_id, G

def remove_subtree(G,node):
    for child in list(nx.neighbors(G,node)):
        try:
            remove_subtree(G,child)
        except nx.NetworkXError:
            #Child has already been removed
            return
    G.remove_node(node)

def conditional_remove_subtrees(G,root,fn):
    """BFS a Graph G. If fn(G,root),remove root
    and all descenant nodes"""
    if not root in G:
        return
    if fn(G,root):
        remove_subtree(G,root)
    else:
        for child in list(nx.neighbors(G,root)):
            conditional_remove_subtrees(G,child,fn)

def is_private(G,node):
    return bool(re.search(r"\._(.+)",node))    
def is_contrib(G,node):
    return bool(re.search(r"\.contrib\.",node))
def is_compiler(G,node):
    return bool(re.search(r"\.compiler\.",node))


def clean_graph(G):
    conditional_remove_subtrees(G,"libraries",is_private)
    conditional_remove_subtrees(G, "libraries", is_contrib)
    conditional_remove_subtrees(G, "libraries", is_compiler)



@click.command()
@click.argument("out_path",type=click.Path())
@click.option("--top_k", type=int)
@click.option("--tokenizer_path",type=click.Path())
def main(out_path,tokenizer_path=None,top_k=50):
    all_libs = nx.DiGraph()
    all_libs.add_node("libraries", name = "Libraries")
    
    top_libraries = load_jsonl("/homes/gws/mikeam/RobustDataScience/data/processed/kaggle_most_common_libraries.jsonl")
    top_libraries = sorted(top_libraries,key=lambda x: x["slug_count"])[-top_k:]
    for lib in tqdm(top_libraries):
        package = importlib.import_module(lib["library"])
        if lib["library"] == "sklearn":
            import_submodules(lib["library"])
        if lib["library"] in ["tensorflow","matplotlib","seaborn","os"]:
            continue
        root, pkg_tree = get_tree_for_pkg_better(package,lib_name=lib["library"], max_depth=4)
        
        if pkg_tree:
            paste_at_root(all_libs,pkg_tree,"libraries")
    
    clean_graph(all_libs)
    nx.write_gpickle(all_libs,os.path.join(out_path,'tree.pickle'))
    
    ## We also want to save names, independently
    all_names = [str(x.get("name"))+"\n" for _,x in all_libs.nodes.data()]
    not_kwargs = [str(x.get("name"))+"\n" for _,x in all_libs.nodes.data() if not x.get("kind") == PackageNode.PARAM]
    with open(os.path.join(out_path,"library_names.txt"),"w") as outfile:
        outfile.writelines(set(all_names))
    
    with open(os.path.join(out_path,"library_names_no_kwargs.txt"),"w") as outfile:
        outfile.writelines(set(not_kwargs))
    
    nx.write_edgelist(all_libs,os.path.join(out_path,"tree.edgelist"), data=False)

    ## Tokenize the Tree
    if tokenizer_path:
        vocab_path = os.path.join(tokenizer_path, "vocab.json")
        merges_path = os.path.join(tokenizer_path, "merges.txt")     
        tokenizer = BartTokenizerFast(vocab_path, merges_path)
        tokenized_graph = TokenizedGraph(all_libs, tokenizer)
        with open(os.path.join(out_path, "tokenized_graph.pickle"), 'wb') as handle:
            pickle.dump(tokenized_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    conda_env = os.environ['CONDA_DEFAULT_ENV']
    # if not conda_env == "KaggleMostFrequent50":
    #     sys.exit("{} environment should be KaggleMostFrequent50 to avoid creating a mess".format(conda_env))
    main()    
