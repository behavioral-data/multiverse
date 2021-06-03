import click
from utils import get_inserted_and_removed, remove_git_chars, is_git_line, load_jsonl, write_jsonl
from tqdm import tqdm

import re
from collections import Counter


import ast

IO_REGEX = r"(Path|DataFrame|open|print|head|((read|save|to|load)_(weights|model|csv|txt|json)))\s*\(.*\)"
PLOT_REGEX = r"(plt\.\W*|plot).*\(.*\)"
FUNCTION_REGEX = r"[\s|=|\W|]*(\w*?)\(.*\)"
ASSIGNMENT_REGEX = r"=\s(.*)"


def diff_is_valid(diff: str):
    inserted, removed = get_inserted_and_removed(diff)
    has_both = len(inserted) != 0 and len(removed) != 0
    
    # Edge case where added lines can become
    # a superset of the removed ones

    
    return has_both
    

def diff_lines_only(diff_filter):
    def wrap_filter(filter_obj,diff):
        new_lines = []
        for line in diff.split("\n"):
            if is_git_line(line):
                no_git_chars = remove_git_chars(line)
                if diff_filter(filter_obj,no_git_chars):
                    continue
            new_lines.append(line)
        
        new_diff = "\n".join(new_lines)
        
        if filter_obj.do_validate:
            if diff_is_valid(new_diff):
                return new_diff
            else:
                return None
        else:
            return new_diff

    return wrap_filter
    


def inserted_removed_asts(diff_filter):
    def wrap_filter(filter_obj,diff):
        """Wraps a filter
        Args:
            diff (String): A diff string in GitHub's format
        Returns:
            String: The new, filtered diff, or None if the diff 
                    should be ignored entirely
        """                
        inserted, removed = get_inserted_and_removed(diff,as_string=True)
        
        try:
            inserted_ast = ast.parse(inserted)
            removed_ast = ast.parse(removed)

        except SyntaxError: 
            return diff #Maybe revisit? Do we want to include diffs that aren't valiud syntax?
        
        if not diff_filter(filter_obj, inserted_ast, removed_ast):
            return diff

    return wrap_filter

def inserted_removed(diff_filter):
    """Handles filters that operate by comparing
       inserted lines with removed lines.
    Args:
        diff_filter ([function]): [A filter that takes a list of inserted
                                  lines and a list of removed lines
                                  and returns true if the diff should be filtered
                                  from the dataser]
    """
    def wrap_filter(filter_obj,diff):
        """Wraps a filter
        Args:
            diff (String): A diff string in GitHub's format
        Returns:
            String: The new, filtered diff, or None if the diff 
                    should be ignored entirely
        """                
        inserted, removed = get_inserted_and_removed(diff)
        if not diff_filter(filter_obj,inserted, removed):
            return diff
    
    return wrap_filter

def all_lines(diff_filter):
    """Handles filters that operate by comparing
       inserted lines with removed lines.
    Args:
        diff_filter ([function]): [A filter that takes a list of inserted
                                  lines and a list of removed lines
                                  and returns true if the diff should be filtered
                                  from the dataser]
    """
    def wrap_filter(filter_obj,diff):
        """Wraps a filter
        Args:
            diff (String): A diff string in GitHub's format
        Returns:
            String: The new, filtered diff, or None if the diff 
                    should be ignored entirely
        """                
        
        if not diff_filter(filter_obj,diff.split("\n")):
            return diff
    
    return wrap_filter

class PositionalArgCollector(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.positional_args = []

    def visit_Call(self,node):
        self.positional_args += [str(x.value) for x in node.args if isinstance(x,ast.Constant)]
        self.positional_args += [str(x.id) for x in node.args if isinstance(x,ast.Name)]
        ast.NodeVisitor.generic_visit(self,node)

def get_positional_args(source: str):
    """Returns a list of all positional 
       arguments in a source string.
    Args:
        source (str): 

    Returns:
        [type]: [description]
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    
class Filters():

    def __init__(self,do_validate=True):
        self.do_validate = do_validate

    def validate_filters(self,filters):
        for f in filters:
            try:
                self.get_filter(f)
            except AttributeError:
                raise click.BadOptionUsage("filters","%s is not an available filter." % f)
        return filters
    
    def diff_is_valid(self, diff: str):
        inserted, removed = get_inserted_and_removed(diff)
        return len(inserted) != 0 and len(removed) != 0
    
    
    def get_filter(self,filter_name):
        return getattr(self,filter_name)

    @diff_lines_only
    def remove_comments(self,line):
        return  (len(line.strip()) > 0 and line.strip()[0] == "#")


    @diff_lines_only
    def remove_whitespace(self,line):
        return len(line.strip()) == 0


    @diff_lines_only
    def remove_io(self,line):
        return bool(re.search(IO_REGEX,line))
    

    @inserted_removed
    def ignore_reshuffles(self,inserted: list, removed: list):
        return set(inserted) == set(removed)

    @inserted_removed
    def no_changes(self,inserted: list, removed: list):
        return len(inserted) > 0 or len(removed) > 0

    @inserted_removed
    def ignore_variable_renames(self,inserted: list, removed: list):
        inserted_assignments = sorted(re.findall(ASSIGNMENT_REGEX, ("\n").join(inserted)))
        removed_assignments = sorted(re.findall(ASSIGNMENT_REGEX, ("\n").join(removed)))
        return inserted_assignments == removed_assignments


    @diff_lines_only
    def remove_plotting(self,line):
        return bool(re.search(PLOT_REGEX, line))


    @inserted_removed_asts
    def remove_positional_arg_diffs(self,inserted_ast, removed_ast):
        inserted_arg_collector = PositionalArgCollector()
        inserted_arg_collector.visit(inserted_ast)
        inserted_args = inserted_arg_collector.positional_args

        removed_arg_collector = PositionalArgCollector()
        removed_arg_collector.visit(removed_ast)
        removed_args = removed_arg_collector.positional_args

        return sorted(inserted_args) != sorted(removed_args)
    

def get_filter(filter_name):
    return getattr(Filters,filter_name)

def get_looks_like_function(diff):
   return re.findall(FUNCTION_REGEX,diff)

def get_functions_from_changed_lines(diff): 
    
    inserted_lines, removed_lines = get_inserted_and_removed(diff, as_string=True)
    changed_lines = inserted_lines + "\n" + removed_lines
    return get_looks_like_function(changed_lines)

def build_function_vocabulary(pairs):
    all_functions = []
    for pair in pairs:
        all_functions += get_functions_from_changed_lines(pair["cell_diff"])
    return Counter(all_functions)

def remove_pairs_without_common_function_calls(pairs, topk=500):
    function_vocab = build_function_vocabulary(pairs).most_common(topk)
    most_common_functions = {x[0] for x in function_vocab}
    new_pairs = []
    for pair in tqdm(pairs, desc="Removing uncommon functions"):
        funcs_from_changed_lines = get_functions_from_changed_lines(pair["cell_diff"])
        if any([x in most_common_functions for x in funcs_from_changed_lines]):
            new_pairs.append(pair)
    return new_pairs

def filter_by_substrings(pairs,substrings_path):
    with open(substrings_path) as f:
        substrings = set([x for x in f.read().splitlines() if len(x)>3])
    new_diffs = []
    matches = []
    for pair in tqdm(pairs, desc = "Removing diffs without substrings"):
        inserted, removed = get_inserted_and_removed(pair["cell_diff"], as_string=True)
        for substr in substrings:
            if substr in inserted or substr in removed:
                matches.append(substr)
                new_diffs.append(pair)
                break
        # if any([x in pair["cell_diff"] for x in substrings]):
        #     new_diffs.append(pair)
    return new_diffs

def filter_diff_size(pairs,size):
    new_pairs = []
    for pair in tqdm(pairs, desc = "Removing max size"):
        inserted, removed = get_inserted_and_removed(pair["cell_diff"])
        if max(len(inserted),len(removed)) < size:
            new_pairs.append(pair)
    return new_pairs

def filter_size(pairs,size):
    new_pairs = []
    for pair in tqdm(pairs, desc = "Removing max size"):
        all_lines = [x for x in pair["cell_diff"].split("\n") if len(x.strip()) > 0]
        if len(all_lines) < size:
            new_pairs.append(pair)
    return new_pairs


def do_clean_git_headers(diffs):
    header_regex = r"--- \n\+\+\+ \n@@\s([+\-][0-9]*,*[0-9]*\s){2}@@\n"
    new_diffs = []
    for diff in tqdm(diffs, desc = "Removing git headers"):
        new_cell_diff = re.sub(header_regex,"",diff["cell_diff"])
        if diff_is_valid(new_cell_diff):
            diff["cell_diff"] = new_cell_diff
            new_diffs.append(diff)
    return new_diffs



@click.argument('pair_path', type=click.Path())
@click.option('--out_path', type=click.Path())
@click.option('--substring_filter', type=click.Path(), help="A list of substrings, one on each line, where each diff contain at least one substring")
@click.option('--filter_names', is_flag=False, default='', show_default=True,type=click.STRING, help='Sets filters to use')
@click.option('--split_diffs', is_flag=True, default=False)
@click.option('--topk_functions', type=click.INT, default=None)
@click.option('--do_validate', is_flag=True, default=False)
@click.option('--max_diff_size', type=click.INT, default=None)
@click.option('--max_size', type=click.INT, default=None)
@click.option('--clean_git_headers', is_flag=True, default=None)
@click.command()  

def main(pair_path, out_path, substring_filter, filter_names, split_diffs, 
        topk_functions, do_validate, max_diff_size, max_size, clean_git_headers):
    
    filters = Filters(do_validate=do_validate)
    
    filter_names = filters.validate_filters([c.strip() for c in filter_names.split(',')])
    pairs = load_jsonl(pair_path)

    filtered_pairs = pairs
    print("Starting with {} pairs".format(len(pairs)))


    if split_diffs:

        split_diffs_on = r"(\n\s*){2,}"
        new_filtered_pairs = []
        for pair in tqdm(filtered_pairs, desc="Splitting diffs"):
            mini_diffs = re.split(split_diffs_on,pair["cell_diff"])
            for diff in mini_diffs:
                if diff_is_valid(diff):
                    new_pair = pair.copy()
                    new_pair["cell_diff"] = diff
                    new_filtered_pairs.append(new_pair)

        print("Expanded {} pairs into {}".format(len(filtered_pairs),len(new_filtered_pairs)))        
        filtered_pairs = new_filtered_pairs
    
    if clean_git_headers:
        len_before = len(filtered_pairs)
        filtered_pairs = do_clean_git_headers(filtered_pairs)
        len_after = len(filtered_pairs)
        print("Removed {} pairs, {} remaining".format(len_before - len_after, len_after))
        
    for f in filter_names:
        filt = filters.get_filter(f)
        new_filtered_pairs = []
        pairs_removed = 0
        lines_removed = 0
        for pair in tqdm(filtered_pairs, desc = f):
            filter_result = filt(pair["cell_diff"])
            if filter_result:
                lines_removed += len(pair["cell_diff"].split("\n")) - len(filter_result.split("\n"))
                pair["cell_diff"] = filter_result
                new_filtered_pairs.append(pair)
            else:
                pairs_removed += 1
        filtered_pairs = new_filtered_pairs
        print("{} pairs removed entirely, {} lines removed, {} remaining".format(pairs_removed,
                                                              lines_removed,                      
                                                              len(new_filtered_pairs)))


    if topk_functions:
        len_before = len(filtered_pairs)
        filtered_pairs = remove_pairs_without_common_function_calls(filtered_pairs,topk=topk_functions)
        len_after = len(filtered_pairs)
        print("Removed {} pairs, {} remaining".format(len_before - len_after, len_after))
   
    if substring_filter:
        len_before = len(filtered_pairs)
        filtered_pairs = filter_by_substrings(filtered_pairs,substring_filter)
        len_after = len(filtered_pairs)
        print("Removed {} pairs, {} remaining".format(len_before - len_after, len_after))

    if max_diff_size:
        len_before = len(filtered_pairs)
        filtered_pairs = filter_diff_size(filtered_pairs,max_diff_size)
        len_after = len(filtered_pairs)
        print("Removed {} pairs, {} remaining".format(len_before - len_after, len_after))

    if max_size:
        len_before = len(filtered_pairs)
        filtered_pairs = filter_size(filtered_pairs, max_size)
        len_after = len(filtered_pairs)
        print("Removed {} pairs, {} remaining".format(len_before - len_after, len_after))

    if out_path:
        print("Saving to {}".format(out_path))
        with open(out_path,"w") as f:
            write_jsonl(f,filtered_pairs)
    

if __name__ == "__main__":  
    main()