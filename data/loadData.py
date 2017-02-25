import gzip
import _pickle as cPickle
import urllib
import os

from os.path import isfile

PREFIX = "./data/"


def load(filename):
    f = gzip.open(filename,'rb')
    return f

def atisfull():
    f = load(PREFIX + 'atis.pkl.gz')
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    f = load(PREFIX + 'atis.fold'+str(fold)+'.pkl.gz')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

def ner():
    dicts = {}
    dicts['labels2idx'] = {}
    dicts['words2idx'] = {}

    for type in ["train", "test", "valid"] :
        with open(PREFIX + "ner." + type) as f:
            for line in f:
                if len(line.strip()) is 0 :
                    continue
                a, b, c, d = line.strip().split()
                a = a.lower()
                if a not in dicts['words2idx']:
                    dicts['words2idx'][a] = len(dicts['words2idx'])
                if d not in dicts['labels2idx']:
                    dicts['labels2idx'][d] = len(dicts['labels2idx'])

    out = {}
    for type in ["train", "test", "valid"] :
        w = []
        l = []
        o = []
        ww = []
        ll = []
        oo = []
        with open(PREFIX + "ner." + type) as f:
            for line in f:
                if len(line.strip()) is 0 :
                    if len(w) > 0:
                        ww.append(w)
                        ll.append(l)
                        oo.append(o)
                        w = []
                        l = []
                        o = []
                    continue
                a, b, c, d = line.strip().split()
                a = a.lower()
                w.append(dicts['words2idx'][a])
                l.append(dicts['labels2idx'][d])
                o.append(0)
        out[type] = (ww, oo, ll)
    out["train"][0].extend(out["valid"][0])
    out["train"][1].extend(out["valid"][1])
    out["train"][2].extend(out["valid"][2])
    return out["train"], out["valid"], out["test"], dicts


def chunk():
    dicts = {}
    dicts['labels2idx'] = {}
    dicts['words2idx'] = {}

    for type in ["train", "test", "valid"] :
        with open(PREFIX + "chunk." + type) as f:
            for line in f:
                if len(line.strip()) is 0 :
                    continue
                a, b, c = line.strip().split()
                a = a.lower()
                if a not in dicts['words2idx']:
                    dicts['words2idx'][a] = len(dicts['words2idx'])
                if c not in dicts['labels2idx']:
                    dicts['labels2idx'][c] = len(dicts['labels2idx'])


    out = {}
    for type in ["train", "test", "valid"] :
        w = []
        l = []
        o = []
        ww = []
        ll = []
        oo = []
        with open(PREFIX + "chunk." + type) as f:
            for line in f:
                if len(line.strip()) is 0 :
                    if len(w) > 0:
                        ww.append(w)
                        ll.append(l)
                        oo.append(o)
                        w = []
                        l = []
                        o = []
                    continue
                a, b, c = line.strip().split()
                a = a.lower()
                w.append(dicts['words2idx'][a])
                l.append(dicts['labels2idx'][c])
                o.append(0)
        out[type] = (ww, oo, ll)
    out["train"][0].extend(out["valid"][0])
    out["train"][1].extend(out["valid"][1])
    out["train"][2].extend(out["valid"][2])
    return out["train"], out["valid"], out["test"], dicts

def pos():
    dicts = {}
    dicts['labels2idx'] = {}
    dicts['words2idx'] = {}

    for type in ["train", "test", "valid"] :
        with open(PREFIX + "chunk." + type) as f:
            for line in f:
                if len(line.strip()) is 0 :
                    continue
                a, b, c = line.strip().split()
                a = a.lower()
                if a not in dicts['words2idx']:
                    dicts['words2idx'][a] = len(dicts['words2idx'])
                if b not in dicts['labels2idx']:
                    dicts['labels2idx'][b] = len(dicts['labels2idx'])

    out = {}
    for type in ["train", "test", "valid"] :
        w = []
        l = []
        o = []
        ww = []
        ll = []
        oo = []
        with open(PREFIX + "chunk." + type) as f:
            for line in f:
                if len(line.strip()) is 0 :
                    if len(w) > 0:
                        ww.append(w)
                        ll.append(l)
                        oo.append(o)
                        w = []
                        l = []
                        o = []
                    continue
                a, b, c = line.strip().split()
                a = a.lower()
                w.append(dicts['words2idx'][a])
                l.append(dicts['labels2idx'][b])
                o.append(0)
        out[type] = (ww, oo, ll)
    return out["train"], out["valid"], out["test"], dicts
