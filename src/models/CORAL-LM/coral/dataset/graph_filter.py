# coding=utf-8
# created by Ge Zhang @ Mar 25, 2020
# filters for graph(code snippets)

KEY_LIBS = ['statsmodels', 'gensim', 'keras',
            'sklearn', 'xgboost', 'scipy', 'pandas']


def key_lib(graph):
    funcs = graph['funcs']
    heads = [f.split('.')[0] for f in funcs]
    if any([h in KEY_LIBS for h in heads]):
        return True
    return False
