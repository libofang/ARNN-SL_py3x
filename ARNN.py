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
        --verbose           verbose, print more text    [default: 2]
        --dataset           the dataset: pos, chunk, ner, atis [default: ner]
        --fold              0,1,2,3,4  used only for atis  [default: 3]
        --h_win_left        (0, 0) for standard RNN    [default: 0]
        --h_win_right       (0, 0) for standard RNN    [default: 0]


        --dyn        Dynamic context windows
        --sub NUM    Subsampling threshold [default: 0]
        --ngram NUM  ngram size [default: 1]
        --cbow NUM  use cbow [default: 0]
        --solid      Use ngram as vectors
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --thr2 NUM   The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
    """)


    params = { # default setting
        'emb_dimension': 100,  # dimension of word embedding
        'nhidden': 100,  # number of hidden units
        'seed': 123,
        'nepochs': 30,
        'dropRate': 0.2,
        'attention': 'general',
        'lvrg': 0,  # 0 for standard RNN, 1 for attetion.
        'rho': numpy.array([100, 50]).astype(numpy.int32), #100,90,80,70,60,50,0 # combining forward and backward layers
        'WVRoot_folder': '/home/lbf/PycharmProjects/WV/'
    }
    params['verbose'] = int(args['--verbose'])
    params['dataset'] = args['--dyn']
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