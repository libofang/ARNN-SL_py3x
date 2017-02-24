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


if __name__ == '__main__':

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
        'WVRoot_folder': '/home/lbf/PycharmProjects/WV/'
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
                elman_combine.run(s)

                break