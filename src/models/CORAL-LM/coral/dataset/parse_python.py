import sys
import json as json
import ast


import os
import random
from tqdm import tqdm
import pdb
import re




################ From py-150k ################
def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s


def parse_file(filename):
    string = read_file_to_string(filename)
    parse_string(string)
    
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



module_map = {}

class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.nest = 0
        self.funcs = []
        self.linenos = []

    def generic_visit(self, node):

        ast.NodeVisitor.generic_visit(self, node)

    def visit_Call(self, node):
        self.nest += 1
        func = self.process_func(node.func)

        # if func != None and self.nest == 1:
        if func != None:
            self.funcs.append(func)
            self.linenos.append(node.lineno)

        self.generic_visit(node)

        self.nest -= 1

    def process_func(self, node):
        if isinstance(node, ast.Attribute):
            prefix = self.process_func(node.value)
            if prefix is None:
                return module_map.get(node.attr, node.attr)
            else:
                return module_map.get(prefix, prefix) + '.' + node.attr
        elif isinstance(node, ast.Name):
            return module_map.get(node.id, node.id)
        elif isinstance(node, ast.Call):
            return self.process_func(node.func)
        else:
            self.generic_visit(node)

    def reset_funcs(self):
        self.funcs = []
        self.linenos = []
        self.nest = 0


class ModuleVisitor(ast.NodeVisitor):

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):

        if isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_map[target.id] = visitor.process_func(
                        node.value.func)
        else:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_map[target.id] = None

        self.generic_visit(node)

    def visit_Import(self, node):
        for module in node.names:
            if module.asname is not None:
                module_map[module.asname] = module.name
            else:
                module_map[module.name] = module.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        top_module = node.module
        for module in node.names:
            if module.asname is None:
                module_map[module.name] = '{}.{}'.format(
                    top_module, module.name)
            else:
                module_map[module.asname] = '{}.{}'.format(
                    top_module, module.name)
        self.generic_visit(node)

    def visit_alias(self, node):
        self.generic_visit(node)





def extract_funcs_from_py(sources, visitor=None):
    """
    sources is file.read() or string
    must pass **sources**
    """
    if not visitor:
        visitor = Visitor()

    funcs = []
    linenos = []
    tree = ast.parse(sources)
    visitor.visit(tree)

    if visitor.funcs != []:
        funcs = visitor.funcs
        linenos = visitor.linenos

    return funcs, linenos


def extract_module(file=None, sources=None, mdvisitor = None):
    """
    sources is file.read()
    must pass **sources**
    """
    # with open(file, 'r') as f:
    #     sources = f.read()
    if not mdvisitor:
        mdvisitor = ModuleVisitor()
    tree = ast.parse(sources)
    mdvisitor.visit(tree)
