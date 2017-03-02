import numpy
import time
import sys
import subprocess
import os
import random
import numpy as np
import math
import theano
import json

from data import loadData
from numpy import ndarray
from rnn import elman_attention

from data import loadData

from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin


def run(params):
    print(params)

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)
    rhoList = numpy.array([100, 50]).astype(numpy.int32)  # 100,90,80,70,60,50,0 # combining forward and backward layers

    # load the dataset
    eval_options = []
    params['measure'] = 'F1score'
    if params['dataset'] == 'atis':
        train_set, valid_set, test_set, dic = loadData.atisfold(params['fold'])
    if params['dataset'] == 'ner':
        train_set, valid_set, test_set, dic = loadData.ner()
    if params['dataset'] == 'chunk':
        train_set, valid_set, test_set, dic = loadData.chunk()
    if params['dataset'] == 'pos':
        train_set, valid_set, test_set, dic = loadData.pos()
        eval_options = ['-r']
        params['measure'] = 'Accuracy'

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    ## :( hack
    # train_lex = train_lex[::100]
    # train_ne = train_ne[::100]
    # train_y = train_y[::100]
    # valid_lex = valid_lex[::100]
    # valid_ne = valid_ne[::100]
    # valid_y = valid_y[::100]
    # test_lex = test_lex[::100]
    # test_ne = test_ne[::100]
    # test_y = test_y[::100]

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    wv = None
    if params['WVFile'] != 'random':

        # load word vector
        wvnp = np.load(params['WVFile'])
        params['emb_dimension'] = len(wvnp[0])

        # load vocab
        with open(params['WVVocabFile']) as f:
            vocab = [line.strip() for line in f if len(line) > 0]
        wi = dict([(a, i) for i, a in enumerate(vocab)])
        wv = numpy.zeros((vocsize + 1, params['emb_dimension']))
        random_v = math.sqrt(6.0 / numpy.sum(params['emb_dimension'])) * numpy.random.uniform(-1.0, 1.0, (params['emb_dimension']))

        miss = 0  # the number of missing words in pre-trained word embeddings
        for i in range(0, vocsize):
            word = idx2word[i]
            if word in wi:
                wv[i] = wvnp[wi[word]]
                # print wvnp[wi[word]]
            else:
                wv[i] = random_v
                miss += 1
        print("missing words rate : ", miss, '/', vocsize)

    best_valid = numpy.zeros(len(rhoList)) - numpy.inf
    best_test = numpy.zeros(len(rhoList)) - numpy.inf

    testMeasureList = [[] for i in range(len(rhoList))]   # this is used for drawing line chart.
    print(testMeasureList)
    # instanciate the model
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])
    rnn = elman_attention.model(nh=params['nhidden'],
                                nc=nclasses,
                                ne=vocsize,
                                de=params['emb_dimension'],
                                attention=params['attention'],
                                h_win=(params['h_win_left'], params['h_win_right']),
                                lvrg=params['lvrg'],
                                wv=wv)

    # train
    for e in range(params['nepochs']):
        # shuffle
        shuffle([train_lex, train_ne, train_y], params['seed'])

        tic = time.time()
        for i in range(nsentences):
            cwords = contextwin(train_lex[i])
            labels = train_y[i]

            nl, aaL = rnn.train(cwords, labels, params['dropRate'], 1)

            # rnn.normalize()
            if params['verbose']:
                sys.stdout.write(('\r[learning] epoch %i >> %2.2f%%' % (
                    e, (i + 1) * 100. / nsentences) +
                                  ('  average speed in %.2f (min) <<' % (
                                      (time.time() - tic) / 60 / (i + 1) * nsentences)) + (' completed in %.2f (sec) <<' % (
                    (time.time() - tic)))))
                sys.stdout.flush()

        print('start test', time.time() / 60)

        print('start pred train', time.time() / 60)
        predictions_train = [[map(lambda varible: idx2label[varible], w) \
                              for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), params['dropRate'], 0, rhoList)]
                             for x in train_lex]

        predictions_test = [[map(lambda varible: idx2label[varible], w) \
                             for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), params['dropRate'], 0, rhoList)]
                            for x in test_lex]

        predictions_valid = [[map(lambda varible: idx2label[varible], w) \
                              for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), params['dropRate'], 0, rhoList)]
                             for x in valid_lex]

        for i_rho in range(len(rhoList)):

            groundtruth_train = [map(lambda x: idx2label[x], y) for y in train_y]
            words_train = [map(lambda x: idx2word[x], w) for w in train_lex]
            groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
            words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
            groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
            words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

            ptrain = [p[i_rho] for p in predictions_train]
            ptest = [p[i_rho] for p in predictions_test]
            pvalid = [p[i_rho] for p in predictions_valid]

            res_train = conlleval(ptrain, groundtruth_train, words_train, folder + '/current.train.txt' + str(i_rho) + str(params['seed']),
                                  eval_options)
            res_test = conlleval(ptest, groundtruth_test, words_test, folder + '/current.test.txt' + str(i_rho) + str(params['seed']), eval_options)
            res_valid = conlleval(pvalid, groundtruth_valid, words_valid, folder + '/current.valid.txt' + str(i_rho) + str(params['seed']),
                                  eval_options)

            print('                                     epoch', e, ' rhoList ', i_rho,
                  '  train p', res_train['p'], 'valid p', res_valid['p'], '  train r', res_train['r'], 'valid r', res_valid['r'],
                  '  train ', params['measure'], res_train['measure'], 'valid ', params['measure'], res_valid['measure'],
                  'best test ', params['measure'], res_test['measure'], ' ' * 20)

            testMeasureList[i_rho].append(res_test['measure'])

            if res_valid['measure'] > best_valid[i_rho]:
                best_valid[i_rho] = res_valid['measure']
                best_test[i_rho] = res_test['measure']

        for i_rho in range(len(rhoList)):  # this is used for drawing line chart.
            print(i_rho, params['dataset'], params['WVModel'], end=' ')
            for v in testMeasureList[i_rho]:
                print(v, end=' ')
            print('')

        for i_rho in range(len(rhoList)):
            print('current best results', rhoList[i_rho], ' ', best_valid[i_rho], '/', best_test[i_rho])

    with open(params['JSONOutputFile'], 'w') as outputFile:
        params['best'] = ndarray.tolist(best_test)
        params['resultListBasedOnEpochs'] = testMeasureList
        res = json.dump(params, outputFile, sort_keys=True, indent=4, separators=(',', ': '))
        print(res)

# json haokan
# merge json information

# 0 ner unknown 79.41 82.03 81.76 83.95 83.48 86.03 81.54 85.22 83.1 85.49 82.96 85.72 82.55 85.66 82.56 85.81 82.68 85.85
# 0 ner unknown 79.94 82.05 82.4 84.07 82.91 85.61 82.84 85.02 82.15 84.65 83.12 85.36 82.27 85.31 82.46 85.29 82.22 85.41
# 0 ner unknown 80.25 81.6 81.93 83.64 83.98 85.35 82.58 84.89 83.22 85.63 83.33 85.62 82.51 85.99 83.01 85.63 83.02 85.78 82.39 85.75
# 0 ner unknown 80.29 81.92 81.69 83.72 82.99 85.4 81.8 84.97 82.66 85.24 82.5 85.9 82.16 85.43 82.65 84.89 82.25 85.65
# 0 ner unknown 78.97 81.95 81.34 83.74 83.33 85.9 81.61 85.05 82.36 84.48 82.11 85.93 82.56 85.51 82.13 85.13 82.32 85.97
#               80.44 81.02 82.66 83.55 83.51 84.24 85.3 85.07 84.35 85.15 84.76 84.94 84.84 85.3 85.59 85.58 85.5 85.54 85.27 85.5 85.3 85.85 85.17 85.33 85.0

