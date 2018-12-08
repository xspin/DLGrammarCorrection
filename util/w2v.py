import logging
from collections import namedtuple, defaultdict
from six.moves import zip as izip
import  pickle as pk
import os
import numpy as np
from util.config import *


# class SentimentPhrase(object):
#     def __init__(self, words, tags, split, sentiment, sentence_id):
#         self.words = words
#         self.tags = tags
#         self.split = split
#         self.sentiment = sentiment
#         self.sentence_id = sentence_id

#     def __str__(self):
#         return '%s %s %s %s %s' % (self.words, self.tags, self.split, self.sentiment, self.sentence_id)

def load_bin_vec(fname, vocab, flag):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    if flag=='google' or flag=="word2vec":
        with open(fname, "rb") as f:
            header = f.readline()
            print("  Header:", header)
            vocab_size, layer1_size = map(int, header.split())
            print ('  vocab size:',vocab_size,'layer1 size:',layer1_size)
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    ch = chr(ord(ch))
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                # if(line%100000==0 or line+1==vocab_size):
                #     print('{}/{} {:.2f}%'.format(line, vocab_size, line/vocab_size*100))
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
    elif flag=='glove':
    ##glove
        with open(fname, 'r', encoding='utf-8') as f:
            vocab_size = 400000
            for i, line in enumerate(f):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_vecs[word] = coefs
                # if(i%10000==0 or i+1==vocab_size):
                #     print('{}/{} {:.2f}%'.format(i, vocab_size, i/vocab_size*100))
    else:
        print("Error: invalid flag:", flag)

    print('  Found %s word vectors.' % len(word_vecs))
    return word_vecs


def add_unknown_words(word_vecs, vocab, k, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_W(word_vecs, word_index, k):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = max(word_index.values())+1
    #word_idx_map = dict()
    W = np.zeros(shape=(vocab_size, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    #i = 1
    W[SVOCAB[UNK]] = np.random.uniform(-np.sqrt(3.0/k), np.sqrt(3.0/k), k)
    for word, idx in word_index.items():
        if idx == UNK: continue
        if word in word_vecs:
            W[idx] = word_vecs[word]
        else:
            W[idx] = np.random.uniform(-np.sqrt(3.0/k), np.sqrt(3.0/k), k)
        #word_idx_map[word] = i
        #i += 1
    return W #, word_idx_map

def data_process(vocab, w2v_file, flag):
    print('Processing corpus with W2V:', w2v_file)
    #max_l = np.max([len(s.words) for s in sentences])
    #print("number of sentences: " + str(len(sentences)))
    print("  vocab size: " + str(len(vocab)))
    #print("max sentence length: " + str(max_l))

    w2v = load_bin_vec(w2v_file, vocab, flag)
    k = len(next(iter(w2v.values())))
    print("  %d words in w2v with dim %d" % (len(w2v), k))
    W = get_W(w2v, vocab, k)
    return W

