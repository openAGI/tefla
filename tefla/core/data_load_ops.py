import os
from glob import glob
import pandas as pd
import numpy as np


def get_image_files(datadir, left_only=False):
    fs = glob('{}/*'.format(datadir))
    if left_only:
        fs = [f for f in fs if 'left' in f]
    return np.array(sorted(fs))


def get_names(files):
    return [x[0:x.rfind('.')] for x in map(lambda f: os.path.basename(f), files)]


def get_labels(names, labels=None, label_file='data/trainLabels.csv',
               per_patient=False):
    try:
        if labels is None:
            labels = pd.read_csv(label_file,
                                 index_col=0).loc[names].values.flatten()
    except Exception:
        labels = np.zeros(shape=[len(names)])

    if per_patient:
        left = np.array(['left' in n for n in names])
        return np.vstack([labels[left], labels[~left]]).T
    else:
        return labels


def get_some_image_files(datadir, n):
    import random
    fs = glob('{}/*'.format(datadir))
    result = []
    for i in xrange(n):
        c1 = random.choice(fs)
        result.append(c1)
        fs.remove(c1)
    return np.array(sorted(result))
