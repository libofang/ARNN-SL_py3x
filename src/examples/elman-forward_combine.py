import numpy
import time
import sys
import subprocess
import os
import random
import numpy as np
import math
import theano
import socket
sys.path.append("../../")

from src.rnn import elman_attention
from src.data import load
from src.metrics.accuracy import conlleval
from src.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':
    def run(s) :
        print(s)
        folder = os.path.basename(__file__).split('.')[0]
        if not os.path.exists(folder): os.mkdir(folder)
        #print folder

        # load the dataset
        eval_options = []
        if s['dataset'] == 'atis':
            train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
        if s['dataset'] == 'ner':
            train_set, valid_set, test_set, dic = load.ner()
        if s['dataset'] == 'chunk':
            train_set, valid_set, test_set, dic = load.chunk()
        if s['dataset'] == 'pos':
            train_set, valid_set, test_set, dic = load.pos()
            eval_options = ['-r']

        idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
        idx2word = dict((k, v) for v, k in dic['words2idx'].items())



        train_lex, train_ne, train_y = train_set
        valid_lex, valid_ne, valid_y = valid_set
        test_lex, test_ne, test_y = test_set

        vocsize = len(dic['words2idx'])
        nclasses = len(dic['labels2idx'])
        nsentences = len(train_lex)

        wv = None
        if 'WVFolderName' in s:
            # load word vector
            # wv = numpy.zeros((vocsize+1, s['emb_dimension']))
            # input = open(s['wv_folder'] + str(s['emb_dimension']), 'r')
            # for line in input:
            #     tokens = line.split(' ')
            #     wv[int(tokens[0])] = [float(tokens[j]) for j in xrange(1, len(tokens) - 1)]

            # load word vector
            wvnp = np.load(s['WVRoot_folder'] + "" + s['WVFolderName'] + "/" + s['model']+".words" + str(s['emb_dimension']) + ".npy")
            # load vocab
            with open(s['WVRoot_folder'] + "" + s['WVFolderName'] + "/" + s['model']+".words" + str(s['emb_dimension']) + ".vocab") as f:
                vocab = [line.strip() for line in f if len(line) > 0]
            wi = dict([(a, i) for i, a in enumerate(vocab)])
            iw = vocab
            wv = numpy.zeros((vocsize + 1, s['emb_dimension']))
            random_v = math.sqrt(6.0 / numpy.sum(s['emb_dimension'])) * numpy.random.uniform(-1.0, 1.0, (s['emb_dimension']))

            miss = 0 # the number of missing words in pre-trained word embeddings
            for i in range(0, vocsize):
                word = idx2word[i]
                if word in wi:
                    wv[i] = wvnp[wi[word]]
                    # print wvnp[wi[word]]
                else:
                    wv[i] = random_v
                    miss += 1
            print("missing words rate : ", miss, '/', vocsize)

        best_valid = numpy.zeros(len(s['rho'])) - numpy.inf
        best_test = numpy.zeros(len(s['rho'])) - numpy.inf

        test_f1List = [[],[],[],[],[],[] ]  # this is used for drawing line chart.


        # instanciate the model
        numpy.random.seed(s['seed'])
        random.seed(s['seed'])
        rnn = elman_attention.model(nh=s['nhidden'],
                                        nc=nclasses,
                                        ne=vocsize,
                                        de=s['emb_dimension'],
                                        attention=s['attention'],
                                        h_win=s['h_win'],
                                        lvrg=s['lvrg'],
                                        wv=wv)



        # train
        for e in range(s['nepochs']):
            # shuffle
            shuffle([train_lex, train_ne, train_y], s['seed'])
            s['ce'] = e
            tic = time.time()
            for i in range(nsentences):
                cwords = contextwin(train_lex[i])
                labels = train_y[i]

                #nl, aaL = rnn.train(cwords, labels, s['dropRate'], 1)

                # rnn.normalize()
                if s['verbose']:
                    sys.stdout.write(('\r[learning] epoch %i >> %2.2f%%' % (
                        e, (i + 1) * 100. / nsentences) +
                                      ('  average speed in %.2f (min) <<' % (
                                          (time.time() - tic) / 60 / (i + 1) * nsentences)) + (' completed in %.2f (sec) <<' % (
                        (time.time() - tic)))))
                    sys.stdout.flush()

            print('start test', time.time() / 60)

            print('start pred train', time.time() / 60)
            predictions_train = [[map(lambda varible: idx2label[varible], w)\
                                  for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), s['dropRate'], 0, s['rho'])]
                                 for x in train_lex]
            groundtruth_train = [map(lambda x: idx2label[x], y) for y in train_y]
            words_train = [map(lambda x: idx2word[x], w) for w in train_lex]

            predictions_test = [[map(lambda varible: idx2label[varible], w)\
                                 for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), s['dropRate'], 0, s['rho'])]
                                for x in test_lex]
            groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
            words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

            predictions_valid = [[map(lambda varible: idx2label[varible], w)\
                                  for w in rnn.classify(numpy.asarray(contextwin(x)).astype('int32'), s['dropRate'], 0, s['rho'])]
                                 for x in valid_lex]
            groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
            words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

            for i_rho in range(len(s['rho'])) :
                ptrain = [p[i_rho] for p in predictions_train]
                ptest = [p[i_rho] for p in predictions_test]
                pvalid = [p[i_rho] for p in predictions_valid]


                res_train = conlleval(ptrain, groundtruth_train, words_train, folder + '/current.train.txt' + str(s['seed']), eval_options)
                res_test = conlleval(ptest, groundtruth_test, words_test, folder + '/current.test.txt' + str(s['seed']), eval_options)
                res_valid = conlleval(pvalid, groundtruth_valid, words_valid, folder + '/current.valid.txt' + str(s['seed']), eval_options)

                print('                                     epoch', e, ' rho ', i_rho, '  train p', res_train[
                    'p'], 'valid p', res_valid[
                    'p'],'  train r', res_train[
                    'r'], 'valid r', res_valid[
                    'r'],'  train F1', res_train[
                    'f1'], 'valid F1', res_valid[
                    'f1'], 'best test F1', res_test['f1'], ' ' * 20)

                test_f1List[i_rho].append(res_test['f1'])

                if res_valid['f1'] > best_valid[i_rho]:
                    best_valid[i_rho] = res_valid['f1']
                    best_test[i_rho] = res_test['f1']
            for i_rho in range(len(s['rho'])) :   # this is used for drawing line chart.
                print(i_rho, s['dataset'], end=' ')
                if s['model'] == 'glove':
                    print(s['WVFolderName'].replace('skip', 'glove'), end=' ')
                else:
                    print(s['WVFolderName'], end=' ')
                for iff1 in test_f1List[i_rho]:
                    print(iff1, end=' ')
                print('')

            for i_rho in range(len(s['rho'])) :
                print(s['rho'][i_rho], ' ', best_valid[i_rho] , '/' , best_test[i_rho])


        print('BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder)  # this is the final result.


if __name__ == '__main__':
    

    if "m.gsic.titech.ac.jp" in socket.gethostname():
        dir_root='/work/alex/data/linguistic/embeddings/Bofang/'
    else:
        dir_root='/home/lbf/PycharmProjects/WV/'

    s = {
        'verbose': 2,
        'dataset' : 'chunk', # pos, chunk, ner, atis
        'fold': 3,  # 5 folds 0,1,2,3,4  used only for atis
        'h_win': (0, 0),    # (0, 0) for standard RNN.
        'emb_dimension': 100,  # dimension of word embedding
        'nhidden': 100,  # number of hidden units
        'seed': 123,
        'nepochs': 30,
        'dropRate': 0.2,
        'attention': 'general',
        'lvrg': 0,  # 0 for standard RNN, 1 for attetion.
        'rho': numpy.array([100, 50]).astype(numpy.int32), #100,90,80,70,60,50,0 # combining forward and backward layers
        'WVRoot_folder': dir_root

    }


    for model in ['glove', 'skip', 'cbow']: #
        for emb_dimension in [25, 50, 100, 250, 500]:
            for WVFolderName in ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
                                 '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_']:

                model = 'skip'
                emb_dimension = 25
                WVFolderName = ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
                                 '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_'][0]

                s['WVFolderName'] = WVFolderName + model
                s['model'] = 'sgns'
                if model == 'glove':
                    s['WVFolderName'] = WVFolderName + 'skip'
                    s['model'] = 'glove'
                s['emb_dimension'] = emb_dimension
                run(s)

                break
