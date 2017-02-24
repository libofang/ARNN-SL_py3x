# ARNN-SL for python3.x

run examples/elman-forward_combine.py file with Environment Variable 
```
THEANO_FLAGS=cuda.root=/usr/local/cuda,device=gpu,floatX=float32
```

See the main function of examples/elman-forward_combine.py file for detailed hyper-parameter settings.

toy pre-trained word embeddings can be found:
embedding: http://202.112.113.8/d/ARNN-SL/WV/201308_p_word_linear-2_skip/sgns.words25.npy    
vocab: http://202.112.113.8/d/ARNN-SL/WV/201308_p_word_linear-2_skip/sgns.words25.vocab


Following script will install the theano-related packages that needed for running the code.
```
sudo apt-get install python3-numpy python3-scipy python3-dev python3-pip python3-nose g++ libopenblas-dev git
sudo pip install Theano
```