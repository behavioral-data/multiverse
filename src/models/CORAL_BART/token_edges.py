import ast
import asttokens
import ast


class CodeTree():
    def __init__(self, source):
        self.source = source
        self.atok = asttokens.ASTTokens(source, parse=True)
        self.tokens = self.atok.tokens
        self.startpos2token = {}
        self.next_start = {}
        self.ast_edges = {}
        for i, t in enumerate(self.tokens):
            self.startpos2token[t.startpos] = t
            self.next_start[t.startpos] = t.endpos if i + \
                1 == len(self.tokens) else self.tokens[i + 1].startpos

        self.token_tree, _ = self.get_token_tree(self.atok.tree)
        for i, tk in enumerate(self.token_tree):
            tk["id"] = i

    def walk_tree_depth_first(self, node):
        '''
        param
            node: ast.Node
        return
            serialized_tree: list[ast.Node]
        '''
        serialized_tree = []
        serialized_tree.append(node)
        if hasattr(node, 'first_token'):
            print(node.first_token, node.first_token.start)
            print(node.last_token, node.last_token.end)

        for n in ast.iter_child_nodes(node):
            serialized_tree += self.walk_tree_depth_first(n)
        return serialized_tree

    def get_token_tree(self, node):
        tree = []
        if hasattr(node, 'first_token'):
            tree.append({"id": None,
                         "span": (node.first_token.startpos, node.last_token.endpos),
                         "children": []})
            idx = len(tree) - 1
        span = (node.first_token.startpos, node.last_token.endpos)
        children_tokens_id = []
        for n in ast.iter_child_nodes(node):
            if not hasattr(n, 'first_token'):
                continue
            sub_tree, sub_span = self.get_token_tree(n)
            tree[idx]["children"].append(sub_span)
            tree += sub_tree
            children_tokens_id += sub_tree[0]["tokens_id"]
        tree[idx]["tokens"] = self.remove_child(span, tree[idx]["children"])
        tree[idx]["tokens_id"] = self.remove_child_id(
            span, tree[idx]["children"])
        for tk in tree[idx]["tokens_id"]:
            self.ast_edges[tk] = children_tokens_id
        if len(tree[idx]["tokens"]) == 0:
            tree = tree[:idx] + tree[idx + 1:]
        return tree, span

    def remove_child(self, span, children):
        start, end = span
        tokens = []
        for child in children:
            c_start, c_end = child
            if start < c_start:
                tokens += self.search_tokens_in_span((start, c_start))
            start = c_end
            while start not in self.startpos2token:
                start += 1
        if start < end:
            tokens += self.search_tokens_in_span((start, end))
        return tokens

    def search_tokens_in_span(self, span):
        tokens = []
        start, end = span
        while(start < end):
            tokens.append(self.startpos2token[start].string)
            start = self.next_start[start]
        return tokens

    def remove_child_id(self, span, children):
        start, end = span
        tokens = []
        for child in children:
            c_start, c_end = child
            if start < c_start:
                tokens += self.search_tokens_in_span_id((start, c_start))
            start = c_end
            while start not in self.startpos2token:
                start += 1

        if start < end:
            tokens += self.search_tokens_in_span_id((start, end))
        return tokens

    def search_tokens_in_span_id(self, span):
        tokens = []
        start, end = span
        while(start < end):
            tokens.append(self.startpos2token[start].index)
            start = self.next_start[start]
        return tokens
