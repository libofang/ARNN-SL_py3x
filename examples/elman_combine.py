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
    if params['WVFolder'] != 'random':
        params['WVFile'] = params['WVFolder'] + '/' + 'words' + str(params['WVModel']['emb_dimension']) + '.npy'
        params['WVVocabFile'] = params['WVFolder'] + '/' + 'words' + str(params['WVModel']['emb_dimension']) + '.vocab'

        # load word vector
        wvnp = np.load(params['WVFile'])
        params['WVModel']['emb_dimension'] = len(wvnp[0])

        # load vocab
        with open(params['WVVocabFile']) as f:
            vocab = [line.strip() for line in f if len(line) > 0]
        wi = dict([(a, i) for i, a in enumerate(vocab)])
        wv = numpy.zeros((vocsize + 1, params['WVModel']['emb_dimension']))
        random_v = math.sqrt(6.0 / numpy.sum(params['WVModel']['emb_dimension'])) * numpy.random.uniform(-1.0, 1.0, (params['WVModel']['emb_dimension']))

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
        params['WVModel']['vocab_size'] = len(vocab)

    print(json.dumps(params,sort_keys=True, indent=4, separators=(',', ': ')))

    rhoSuffix = "%_forward"
    best_valid = {}
    best_test = {}
    for i_rho in range(len(rhoList)):
        best_valid[str(rhoList[i_rho]) + rhoSuffix] = -numpy.inf
        best_test[str(rhoList[i_rho]) + rhoSuffix] = -numpy.inf
    validMeasureList = {}
    testMeasureList = {}   # this is used for drawing line chart.
    for i_rho in range(len(rhoList)):
        validMeasureList[str(rhoList[i_rho]) + rhoSuffix] = []
        testMeasureList[str(rhoList[i_rho]) + rhoSuffix] = []

    # instanciate the model
    numpy.random.seed(params['seed'])
    random.seed(params['seed'])
    rnn = elman_attention.model(nh=params['nhidden'],
                                nc=nclasses,
                                ne=vocsize,
                                de=params['WVModel']['emb_dimension'],
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

            validMeasureList[str(rhoList[i_rho]) + rhoSuffix].append(res_valid['measure'])
            testMeasureList[str(rhoList[i_rho]) + rhoSuffix].append(res_test['measure'])

            if res_valid['measure'] > best_valid[str(rhoList[i_rho]) + rhoSuffix]:
                best_valid[str(rhoList[i_rho]) + rhoSuffix] = res_valid['measure']
                best_test[str(rhoList[i_rho]) + rhoSuffix] = res_test['measure']

        for i_rho in range(len(rhoList)):  # this is used for drawing line chart.
            print(i_rho, params['dataset'], params['WVModel'], end=' ')
            for v in testMeasureList[str(rhoList[i_rho]) + rhoSuffix]:
                print(v, end=' ')
            print('')

        for i_rho in range(len(rhoList)):
            print('current best results', rhoList[i_rho], ' ', best_valid[str(rhoList[i_rho]) + rhoSuffix], '/', best_test[str(rhoList[i_rho]) + rhoSuffix])

    with open(params['JSONOutputFile'], 'w') as outputFile:
        params['results'] = {}
        params['results']['best_valid_' + params['measure']] = best_valid
        params['results']['best_test_' + params['measure']] = best_test
        params['results']['valid_' + params['measure'] + 'ListBasedOnEpochs'] = validMeasureList
        params['results']['test_' + params['measure'] + 'ListBasedOnEpochs'] = testMeasureList

        res = json.dump(params, outputFile, sort_keys=True, indent=4, separators=(',', ': '))
        print(res)

# json haokan
# merge json information

