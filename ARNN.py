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


if __name__ == '__main__':

    args = docopt("""
    Usage:
        ARNN.py [options]

    Options:
        --verbose NUM       verbose print more text    [default: 2]
        --dataset STRING    the dataset: pos, chunk, ner, atis [default: ner]
        --fold NUM          0,1,2,3,4  used only for atis dataset  [default: 3]
        --h_win_left NUM    0 for standard RNN    [default: 0]
        --h_win_right NUM   0 for standard RNN    [default: 0]
    """)


    params = { # default setting
        'emb_dimension': 100,  # dimension of word embedding
        'nhidden': 100,  # number of hidden units
        'seed': 123,
        'nepochs': 30,
        'dropRate': 0.0,
        'attention': 'general',
        'lvrg': 0,  # 0 for standard RNN, 1 for attetion.
        'rho': numpy.array([100, 50]).astype(numpy.int32), #100,90,80,70,60,50,0 # combining forward and backward layers
        'WVRoot_folder': '/home/lbf/PycharmProjects/WV/'
    }
    params['verbose'] = int(args['--verbose'])
    params['dataset'] = args['--dataset']
    params['h_win'] = (int(args['--h_win_left']), int(args['--h_win_right']))

    for model in ['glove', 'skip', 'cbow']: #
        for emb_dimension in [25, 50, 100, 250, 500]:
            for WVFolderName in ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
                                 '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_']:

                model = 'skip'
                emb_dimension = 25
                WVFolderName = ['201308_p_word_linear-2_' , '201308_p_structured_linear-2_' ,
                                 '201308_p_word_dependency-1_', '201308_p_structured_dependency-1_'][0]

                params['WVFolderName'] = WVFolderName + model
                params['model'] = 'sgns'
                if model == 'glove':
                    params['WVFolderName'] = WVFolderName + 'skip'
                    params['model'] = 'glove'
                params['emb_dimension'] = emb_dimension
                elman_combine.run(params)

                break