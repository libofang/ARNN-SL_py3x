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
import time
from datetime import datetime


if __name__ == '__main__':


    # temp
    # for folderName in os.listdir("../WV/201308_p/"):
    #     print("mv " + folderName + " " + folderName.replace("2_", '2/').replace('1_', '1/'))
    # for folderName in os.listdir("../WV/201308_p/"):
    #     for fileName in os.listdir("../WV/201308_p/" + folderName + "/"):
    #         print("mv " + folderName + "/" + fileName + " " + folderName + "/" + fileName.replace("sgns.", '').replace('glove.', ''))
    args = docopt("""
    Usage:
        ARNN.py [options] <WVFolder> <JSONOutputFile>



    Options:
        --verbose NUM           verbose print more text    [default: 2]
        --RNNModel STRING       currently only support (bi-) LSTM [default: LSTM]
        --dataset STRING        the dataset: pos, chunk, ner [default: chunk]
        --fold NUM              0,1,2,3,4  used only for atis dataset  [default: 3]
        --h_win_left NUM        0 for standard RNN    [default: 0]
        --h_win_right NUM       0 for standard RNN    [default: 0]
        --h_win NUM             -1 indicate the h_win_left/right paramter should be used. [default: -1]
        --nhidden NUM           number of hidden units  [default: 100]
        --seed NUM              ramdom seed     [default: 12]
        --nepochs NUM           number of epochs [default: 20]
        --dropRate NUM          drop rate [default: 0.0]
        --attention STRING      attention type general/concat [default: general]
        --lvrg NUM              leverage the impact of hidden layer and attention opponent, 0 for standard RNN, 1 for attention [default: 0]

        --emb_dimension NUM     dimension of word embedding [default: 500]

    """)

    #--WVFile FILE           word vector file, npy format. 'random' indicate randomly initialized word vectors [default: ../WV/201308_p_word_linear-2_skip/sgns.words100.npy]
    #--WVVocabFile FILE      word vector vocab file [default: ../WV/201308_p_word_linear-2_skip/sgns.words100.vocab]


    params = {} # all the parameters
    params['WVModel'] = {} # this param is deduced from the file/folder name

    params['verbose'] = int(args['--verbose'])
    params['dataset'] = args['--dataset']
    params['fold'] = int(args['--fold'])
    params['h_win_left'] = int(args['--h_win_left'])
    params['h_win_right'] = int(args['--h_win_right'])
    params['h_win'] = int(args['--h_win'])
    if params['h_win'] != -1:
        params['h_win_left'] = params['h_win']
        params['h_win_right'] = params['h_win']

    params['nhidden'] = int(args['--nhidden'])
    params['seed'] = int(args['--seed'])
    params['nepochs'] = int(args['--nepochs'])
    params['dropRate'] = float(args['--dropRate'])
    params['attention'] = args['--attention']
    params['lvrg'] = int(args['--lvrg'])

    params['WVModel']['emb_dimension'] = int(args['--emb_dimension'])

    params['WVFolder'] = args['<WVFolder>']
    params['JSONOutputFile'] = args['<JSONOutputFile>']


    # deduce the WVModel from the file/folder name
    if params['WVFolder'] == 'random':
        params['WVModel']['model'] = 'random'
    else:
        modelIndex = 0

        if '[' in params['WVFolder'] and ']' in params['WVFolder']:
            folderSet = set(eval(params['WVFolder'].replace('[', '[\'').replace(']', '\']').replace(',', '\',\'')))
        else:
            folderSet = [params['WVFolder']]
        for folder in folderSet:
            params['WVModel' + str(modelIndex)] = {}
            if 'skip' in folder:
                params['WVModel' + str(modelIndex)]['model'] = 'skip'
                params['WVModel' + str(modelIndex)]['iteration'] = 2
                params['WVModel' + str(modelIndex)]['negative_sampling'] = 5
            if 'cbow' in folder:
                params['WVModel' + str(modelIndex)]['model'] = 'cbow'
                params['WVModel' + str(modelIndex)]['iteration'] = 5
                params['WVModel' + str(modelIndex)]['negative_sampling'] = 5
            if 'glove' in folder:
                params['WVModel' + str(modelIndex)]['model'] = 'glove'
                params['WVModel' + str(modelIndex)]['iteration'] = 30
            params['WVModel' + str(modelIndex)]['context'] = {}
            if 'structured' in folder:
                params['WVModel' + str(modelIndex)]['context']['representation'] = 'bound'
            else:
                params['WVModel' + str(modelIndex)]['context']['representation'] = 'unbound'
            if 'dependency' in folder:
                params['WVModel' + str(modelIndex)]['context']['type'] = 'dependency-based'
                params['WVModel' + str(modelIndex)]['window'] = folder.split('dependency-')[1].split('/')[0]
            else:
                params['WVModel' + str(modelIndex)]['context']['type'] = 'linear'
                params['WVModel' + str(modelIndex)]['window'] = folder.split('linear-')[1].split('/')[0]
            if '201308_p' in folder:
                params['WVModel' + str(str(modelIndex))]['corpus'] = {}
                params['WVModel' + str(modelIndex)]['corpus'] = 'wikipedia 201308 dump'
            else:
                params['WVModel' + str(modelIndex)]['corpus'] = 'unknown'
            params['WVModel' + str(modelIndex)]['min_count'] = 100
            modelIndex = modelIndex + 1

    elman_combine.run(params)


    # for model in ['glove', 'skip', 'cbow']: #
    #     for emb_dimension in [25, 50, 100, 250, 500]:
    #         for WVFolderName in ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
    #                              '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_']:
    #
    #             model = 'skip'
    #             emb_dimension = 25
    #             WVFolderName = ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
    #                              '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_'][0]
    #
    #             params['WVFolderName'] = WVFolderName + model
    #             params['model'] = 'sgns'
    #             if model == 'glove':
    #                 params['WVFolderName'] = WVFolderName + 'skip'
    #                 params['model'] = 'glove'
    #             params['emb_dimension'] = emb_dimension
    #             elman_combine.run(params)
    #
    #             break