import os
from collections import defaultdict
from util.config import *
import util.w2v as w2v
import numpy as np
import nltk.tokenize as tok
import re
import pickle as pk


def text_process(text):
    text = text.lower()
    sentences = tok.sent_tokenize(text)
    sent_token = list(map(tok.word_tokenize, sentences))
    # sent_token = list(filter(lambda x:len(x)<=mlen, sent_token))
    return sent_token

def get_type(tokens):
    tp = []
    idx = []
    verb_regex = r'[a-z]+(ing|ed|en)$'
    for i,word in enumerate(tokens):
        t = None
        if word in vocab['prep']: t = 'prep'
        elif word in vocab['art']: t = 'art'
        elif word in vocab['pred']: t = 'pred'
        elif word in vocab['punc']: t = 'punc'
        elif re.match(verb_regex, word): t = 'verb'
        tp.append(t)
        if t: idx.append(i)
    return tp, idx

def add_noise(tokens, type=None):
    '''add noise to the sentence randomly choosing the following 6 erroneous types
        - preposition
        - article
        - predicate
        - punctuation
        - noun
        - verb
    '''
    # types = ['prep', 'noun', 'article', 'verb', 'pred', 'punc']
    if np.random.rand()<0.1: return
    types, idx = get_type(tokens)

    # randomly switch 2 letters 
    # if np.random.rand()<0.01 or (idx==[] and np.random.rand()<0.1): 
    #     i = np.random.randint(len(tokens))
    #     if len(tokens[i])>2: 
    #         j = np.random.randint(len(tokens[i])-1)
    #         tokens[i] = tokens[i][:j] + tokens[i][j+1] + tokens[i][j] + tokens[i][j+2:]
    #     return

    if idx == []: return
    i = np.random.choice(idx)
    tp = types[i]
    if tp in ['prep', 'art', 'pred', 'punc']:
        if np.random.rand() < 0.1: del tokens[i]
        else: tokens[i] = np.random.choice(vocab[tp])
    elif tp == 'verb':
        if tokens[i][-3:] == 'ing': tokens[i] = tokens[i][:-3]+'ed'
        elif tokens[i][-2:] == 'ed': 
            if np.random.rand() < 0.5: tokens[i] = tokens[i][:-2]+'en'
            else: tokens[i] = tokens[i][:-2]+'ing'
        elif tokens[i][-2:] == 'en':  tokens[i] = tokens[i][:-2]+'ed'

def vocab_process(texts, vocab_size, embed):
    word_dict = defaultdict(lambda: 0)
    for text in texts:
        for w in text: word_dict[w] += 1
    word_dict = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
    print('  word counts: {}, vocab size: {}'.format(len(word_dict), vocab_size))
    word_index = {}
    vocab_size -= len(SVOCAB)
    for w,i in SVOCAB.items(): word_index[w] = i
    for i in range(min(len(word_dict),vocab_size)): word_index[word_dict[i][0]] = i+len(SVOCAB)
    for i in range(vocab_size, len(word_dict)): word_index[word_dict[i][0]] = SVOCAB[UNK]
    # for w in word_dict:
    #     assert w in word_index, 'error key: %s'%w
    embedmat = w2v.data_process(word_index, embedding_path[embed], embed)
    return word_index, embedmat

def sequence_process(texts, word_index):
    seq = [[word_index[w] for w in text] for text in texts]
    return seq

def stat(seq):
    length = list(map(len, seq))
    return np.max(length), np.min(length), np.mean(length), np.std(length)

def process(name, load_data, embed, regen, character_dim):
    '''process the text data with embedding

    Args:
        load_data: A function that loads data from the original sources and returns sources, targets, src_vocab_size, tgt_vocab_size
        pkfile: the pickle file path to dump the processed data
        embed: embedding method: glove, google, random
        regen: a boolean parameter to indicates if re-processing the origin data
        character_dim: None or a integer
    Returns:
        data = (src_sequences, tgt_sequences, src_word_index, tgt_word_index, src_embedmat, tgt_embedmat, others)
    '''
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    assert embed=='glove' or embed=='google' or embed=='word2vec', "invalid embedding method: "+embed
    if embed=='word2vec': embed = 'google'
    pkfile= os.path.join(pk_dir, 'data_' + name + '_' + embed);
    if character_dim!=None and character_dim>0: pkfile += '_'+str(character_dim)
    pkfile += '.p'
    data = []
    print('Process dataset:', name.upper())
    if not os.path.exists(pk_dir):
        os.mkdir(pk_dir)
    if os.path.exists(pkfile) and not regen:
        print('loading dump file ...')
        data = pk.load(open(pkfile,'rb'))
        src_sequences, tgt_sequences, src_word_index, tgt_word_index,\
            src_embedmat, tgt_embedmat, others = data
    else:
        print('loading data ...')
        # In inputs and outputs, the sentences are tokenizered
        sources, targets, src_vocab_size, tgt_vocab_size, others = load_data()
        print('processing vocab ...')
        src_word_index, src_embedmat = vocab_process(sources, src_vocab_size, embed)
        tgt_word_index, tgt_embedmat = vocab_process(targets, tgt_vocab_size, embed)
        src_sequences = sequence_process(sources, src_word_index)
        tgt_sequences = sequence_process(targets, tgt_word_index)
        data = (src_sequences, tgt_sequences, src_word_index, tgt_word_index, src_embedmat, tgt_embedmat, others)
        print("dump data to", pkfile)
        pk.dump(data, open(pkfile, "wb"))
    print('  src: max/min/mean/std: {:.0f}/{:.0f}/{:.2f}/{:.2f}'.format(*stat(src_sequences)))
    print('  tgt: max/min/mean/std: {:.0f}/{:.0f}/{:.2f}/{:.2f}'.format(*stat(tgt_sequences)))
    print('  src embedding: {}'.format(src_embedmat.shape))
    print('  tgt embedding: {}'.format(tgt_embedmat.shape))
    return data