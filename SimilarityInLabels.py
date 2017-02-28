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
import json


if __name__ == '__main__':

    args = docopt("""
    Usage:
        SimilarityInLabels.py [options] <WVRoot>


    Options:
        --fold NUM              0,1,2,3,4  used only for atis dataset  [default: 3]

        --emb_dimension NUM     dimension of word embedding, -1 indicates using the default dimension in word vector file. [default: -1]
        --WVModel STRING        specify which word embedding model is used [default: unknown]
    """)

    params = {}  # all the parameters


    params['emb_dimension'] = int(args['--emb_dimension'])
    params['WVModel'] = args['--WVModel']

    params['WVRoot'] = args['<WVRoot>']

    resultMap = {}
    for dataset in ['ner']: # , 'pos', 'chunk'
        params['dataset'] = dataset

        wvnpList = []
        wiList = []
        iwList = []
        wvNameList = []
        # load word vector
        for WVFolderName in ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
                             '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_']:
            model = 'skip'
            emb_dimension = params['emb_dimension']

            params['WVFolderName'] = WVFolderName + model
            params['model'] = 'sgns'
            if model == 'glove':
                params['WVFolderName'] = WVFolderName + 'skip'
                params['model'] = 'glove'
            params['emb_dimension'] = emb_dimension

            params['WVFile'] = params['WVRoot'] + params['WVFolderName'] + "/" + params['model'] + ".words" + str(emb_dimension) + ".npy"
            params['WVVocabFile'] = params['WVRoot'] + params['WVFolderName'] + "/" + params['model'] + ".words" + str(emb_dimension) + ".vocab"

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

            wvnpList.append(wvnp)
            wiList.append(wi)
            iwList.append(iw)
            wvNameList.append(WVFolderName + model)


        #load dataset
        print("loading dataset")
        if params['dataset'] == 'ner':
            train_set, valid_set, test_set, dic = loadData.ner()
        if params['dataset'] == 'chunk':
            train_set, valid_set, test_set, dic = loadData.chunk()
        if params['dataset'] == 'pos':
            train_set, valid_set, test_set, dic = loadData.pos()

        idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
        idx2word = dict((k, v) for v, k in dic['words2idx'].items())

        train_lex, train_ne, train_y = train_set
        valid_lex, valid_ne, valid_y = valid_set
        test_lex, test_ne, test_y = test_set


        # finally, some real thing start
        print("calculating labelMap")
        labelMap = {}
        for lexList, neList, yList in zip(*train_set):
            for lex, y in zip(lexList, yList):
                word = idx2word[lex]
                label = idx2label[y]

                inWi = True
                for wi in wiList:
                    if word not in wi: # ignore missing words in embeddings
                        inWi = False
                if not inWi:
                    continue
                label = label.replace('I-', '')
                label = label.replace('B-', '')
                # print(word, label)
                if label not in labelMap:
                    labelMap[label] = set()
                labelMap[label].add(word)

        if 'O' in labelMap:
            del labelMap['O']
        #del labelMap['NP']

        print("calculating distance")

        result = {}
        for k, v in labelMap.items():

            result[k] = {}
            for wvnp, wi, iw, wvName in zip(wvnpList, wiList, iwList, wvNameList):
                sum = 0.0
                count = 0
                for w1 in v:
                    for w2 in v:
                        if w1 == w2:
                            continue
                        sum += wvnp[wi[w1]].dot(wvnp[wi[w2]])
                        count += 1

                print(k, len(v),)
                print(sum)
                if count == 0:
                    continue

                result[k][wvName] = {}
                result[k][wvName]['wordCount'] = len(v)
                result[k][wvName]['similarity'] = sum / count
        print(result)
        resultMap[params['dataset']] = result
    print()
    print(resultMap)
    with open("./simInLabel.txt", 'w') as outputFile:
        json.dump(resultMap, outputFile, sort_keys=True, indent=4, separators=(',', ': '))






