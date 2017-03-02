# ARNN-SL for python3.x

## ARNN

run `python ARNN.py -h` for more detailed hyperparameter settings.
For example, 
```
ARNN.py [options] <WVFile> <WVVocabFile> <JSONOutputFile>
python3 ARNN.py --nepochs 1 ../WV/201308_p_word_linear-2_skip/sgns.words100.npy ../WV/201308_p_word_linear-2_skip/sgns.words100.vocab ./json_output.txt
```
set the THEANO Environment Variable as
```
THEANO_FLAGS=cuda.root=/usr/local/cuda,device=gpu,floatX=float32
```


toy pre-trained word embeddings can be found:
embedding: http://202.112.113.8/d/ARNN-SL/WV/201308_p_word_linear-2_skip/sgns.words25.npy    
vocab: http://202.112.113.8/d/ARNN-SL/WV/201308_p_word_linear-2_skip/sgns.words25.vocab



Following script will install the theano-related packages that needed for running the code.
```
sudo apt-get install python3-numpy python3-scipy python3-dev python3-pip python3-nose g++ libopenblas-dev git
sudo pip install Theano
```


##calculate the average similarity of each word pairs with the same label.

label indicate named entity or pos tag.

Run following:
```
python3 DistanceInLabels.py ../WV/201308_p_word_linear-2_skip/sgns.words100.npy ../WV/201308_p_word_linear-2_skip/sgns.words100.vocab ./json_output.txt
```