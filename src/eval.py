## some code in this file is based on https://github.com/CRIPAC-DIG/GRACE/blob/master/eval.py
import numpy as np
import functools

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from torch import take


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


def label_classification(embeddings, y, ratio = None, train_mask = None, test_mask = None, val_mask = None):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)

    X = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = X[train_mask], X[test_mask], Y[train_mask], Y[test_mask]

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=500).fit(X_train,  y_train.ravel() )
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    y_test = onehot_encoder.transform(y_test).toarray().astype(np.bool)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score( y_test, y_pred, )

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        "acc" : acc
    }
