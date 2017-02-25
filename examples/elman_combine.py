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
    rho = numpy.array([100, 90, 80, 60, 50, 0]).astype(numpy.int32)  # 100,90,80,70,60,50,0 # combining forward and backward layers

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
    train_lex = train_lex[::100]
    train_ne = train_ne[::100]
    train_y = train_y[::100]
    valid_lex = valid_lex[::100]
    valid_ne = valid_ne[::100]
    valid_y = valid_y[::100]
    test_lex = test_lex[::100]
    test_ne = test_ne[::100]
    test_y = test_y[::100]

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    wv = None
    if params['WVFile'] is not 'random':

        # load word vector
        wvnp = np.load(params['WVFile'])
        params['emb_dimension'] = len(wvnp[0])
        # load vocab
        with open(params['WVVocabFile']) as f:
            vocab = [line.strip() for line in f if len(line) > 0]
        wi = dict([(a, i) for i, a in enumerate(vocab)])
        iw = vocab
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

    best_valid = numpy.zeros(len(rho)) - numpy.inf
    best_test = numpy.zeros(len(rho)) - numpy.inf

    test_f1List = [[], [], [], [], [], []]  # this is used for drawing line chart.

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
                              for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), params['dropRate'], 0, rho)]
                             for x in train_lex]

        predictions_test = [[map(lambda varible: idx2label[varible], w) \
                             for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), params['dropRate'], 0, rho)]
                            for x in test_lex]

        predictions_valid = [[map(lambda varible: idx2label[varible], w) \
                              for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), params['dropRate'], 0, rho)]
                             for x in valid_lex]

        for i_rho in range(len(rho)):

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

            print('                                     epoch', e, ' rho ', i_rho, '  train p', res_train[
                'p'], 'valid p', res_valid[
                      'p'], '  train r', res_train[
                      'r'], 'valid r', res_valid[
                      'r'], '  train F1', res_train[
                      'f1'], 'valid F1', res_valid[
                      'f1'], 'best test F1', res_test['f1'], ' ' * 20)

            test_f1List[i_rho].append(res_test['f1'])

            if res_valid['f1'] > best_valid[i_rho]:
                best_valid[i_rho] = res_valid['f1']
                best_test[i_rho] = res_test['f1']

        for i_rho in range(len(rho)):  # this is used for drawing line chart.
            print(i_rho, params['dataset'], params['WVModel'], end=' ')
            for iff1 in test_f1List[i_rho]:
                print(iff1, end=' ')
            print('')

        for i_rho in range(len(rho)):
            print('Best results right now', rho[i_rho], ' ', best_valid[i_rho], '/', best_test[i_rho])

    with open(params['JSONOutputFile'], 'w') as file:
        params['bestF1score'] = ndarray.tolist(best_test)
        params['F1scoreListBasedOnEpochs'] = test_f1List
        jsonResults = json.dumps(params)
        file.write(jsonResults)

# 0.0 bug
# json haokan
# merge json information
