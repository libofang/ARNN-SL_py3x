import numpy
import time
import sys
import subprocess
import os
import random
import numpy as np
import math
import theano

from examples import elman_combine
from docopt import docopt
from numpy import ndarray
from data import loadData


if __name__ == '__main__':

    args = docopt("""
    Usage:
        DistanceInLabels.py [options] <WVFile> <WVVocabFile> <JSONOutputFile>


    Options:
        --dataset STRING        the dataset: pos, chunk, ner [default: ner]
        --fold NUM              0,1,2,3,4  used only for atis dataset  [default: 3]

        --emb_dimension NUM     dimension of word embedding, -1 indicates using the default dimension in word vector file. [default: -1]
        --WVModel STRING        specify which word embedding model is used [default: unknown]

    """)

    params = {}  # all the parameters

    params['dataset'] = args['--dataset']
    params['fold'] = int(args['--fold'])

    params['emb_dimension'] = int(args['--emb_dimension'])
    params['WVModel'] = args['--WVModel']

    params['WVFile'] = args['<WVFile>']
    params['WVVocabFile'] = args['<WVVocabFile>']
    params['JSONOutputFile'] = args['<JSONOutputFile>']

    #load dataset
    print("loading dataset")
    if params['dataset'] == 'atis':
        train_set, valid_set, test_set, dic = loadData.atisfold(params['fold'])
    if params['dataset'] == 'ner':
        train_set, valid_set, test_set, dic = loadData.ner()
    if params['dataset'] == 'chunk':
        train_set, valid_set, test_set, dic = loadData.chunk()
    if params['dataset'] == 'pos':
        train_set, valid_set, test_set, dic = loadData.pos()
        params['measure'] = 'Accuracy'

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    # load word vector
    print("loading word vectors")
    wvnp = np.load(params['WVFile'])

    #           normalization
    norm = np.sqrt(np.sum(wvnp * wvnp, axis=1))
    wvnp = wvnp / norm[:, np.newaxis]

    params['emb_dimension'] = len(wvnp[0])
    #       load vocab
    with open(params['WVVocabFile']) as f:
        vocab = [line.strip() for line in f if len(line) > 0]
    wi = dict([(w, i) for i, w in enumerate(vocab)])
    iw = dict([(i, w) for i, w in enumerate(vocab)])

    # finally, some real thing start
    print("calculating labelMap")
    labelMap = {}
    for lexList, neList, yList in zip(*train_set):
        for lex, y in zip(lexList, yList):
            word = idx2word[lex]
            label = idx2label[y]

            if word not in wi: # ignore missing words in embeddings
                continue

            label = label.replace('I-', '')
            label = label.replace('B-', '')
            # print(word, label)
            if label not in labelMap:
                labelMap[label] = set()
            labelMap[label].add(word)

    del labelMap['O']
    #del labelMap['NP']

    print("calculating distance")
    for k, v in labelMap.items():
        print(k, len(v), )
        sum = 0.0
        for w1 in v:
            for w2 in v:
                if w1 == w2:
                    continue
                sum += wvnp[wi[w1]].dot(wvnp[wi[w2]])

        print(sum)







