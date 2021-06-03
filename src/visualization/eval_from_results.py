import click
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import numpy as np
import json


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

@click.command()
@click.argument("path", type=click.Path())
def main(path):
    results = load_jsonl(path)
    y_true = np.concatenate([x['classification_labels'] for x in results])
    mask = y_true!= -100

    y_prob = softmax(np.concatenate([x["classification_logits"] for x in results])[:,-1])[mask]
    print(y_prob.shape)
    print(y_true.shape)
    print(roc_auc_score(y_true[mask],y_prob))

if __name__ == "__main__":
    main()