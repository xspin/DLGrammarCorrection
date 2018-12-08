from util import w2v 
import os, re
import pickle as pk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import sys

MAX_WORD_LEN = 20

def get_character_embedding(alphabet, dim):
    embedding_matrix = np.zeros([len(alphabet)+1, dim], dtype='float32')
    character_index = {c : i+1 for i,c in enumerate(alphabet)}
    for c in alphabet:
        embedding_matrix[character_index[c]] = np.random.uniform(-np.sqrt(3.0/dim), np.sqrt(3.0/dim), dim)
    return embedding_matrix, character_index
    
def process(name, load_data, embed, regen, character_dim):
    '''process the text data with embedding

    Args:
        load_data: A function that loads data from the original sorces and returns texts, labels, max_nb_words
        pkfile: the pickle file path to dump the processed data
        embed: embedding method: glove, google, random
        regen: a boolean parameter to indicates if re-processing the origin data
        character_dim: None or a integer
    Returns:
        data = (samples, labels, labels_index, word_index, embedmat, nbs)
    '''
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    assert embed=='glove' or embed=='google' or embed=='word2vec', "invalid embedding method: "+embed
    if embed=='word2vec': embed = 'google'
    pkfile='datasets/data_' + name + '_' + embed;
    if character_dim!=None and character_dim>0: pkfile += '_'+str(character_dim)
    pkfile += '.p'
    emb_file = {}
    emb_file['glove'] = "/home/i/Code/python/CRNLP/datasets/embedding/glove.6B.100d.txt"
    emb_file['google'] = "/home/i/Code/python/CRNLP/datasets/embedding/GoogleNews-vectors-negative300.bin"
    data = []
    print('Process dataset:', name.upper())
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    if os.path.exists(pkfile) and not regen:
        print('loading dump file ...')
        data = pk.load(open(pkfile,'rb'))
    else:
        print('loading data ...')
        temp = load_data()
        if len(temp)==4:
            texts, labels, max_nb_words, nbs = temp
            labels_index = {lb:i for i,lb in enumerate(set(labels))}
            labels = [labels_index[lb] for lb in labels]

            # 文本向量化
            tokenizer = Tokenizer(num_words=max_nb_words)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)

            word_index = tokenizer.word_index
            print('Found %s unique tokens.' % len(word_index))

            samples = pad_sequences(sequences, maxlen=max_nb_words)
            # labels = to_categorical(np.asarray(labels))
            labels = np.asarray(labels)
        else:
            samples, labels, word_index, max_nb_words, nbs = temp
            labels_index = {lb:i for i,lb in enumerate(set(labels))}
        print('Shape of data tensor:', samples.shape)
        print('Shape of label tensor:', labels.shape)

        print('word embedding ...')
        embedmat = w2v.data_process(word_index, emb_file[embed], embed)
        embedmat_char = np.array([])
        if character_dim > 0:
            print('character embedding ....')
            wordlencount = {}
            max_word_len = 0
            for word in word_index.keys():
                max_word_len = max(max_word_len, len(word))
            print('max word length:', max_word_len)
            max_word_len = min(max_word_len, MAX_WORD_LEN)
            #embedmat_char, char_index = get_character_embedding(alphabet, character_dim)
            dim = character_dim
            embedvec = {c:np.random.uniform(-np.sqrt(3.0/dim), np.sqrt(3.0/dim), dim) for c in alphabet}
            embedmat_char = np.zeros([embedmat.shape[0], max_word_len, dim], dtype='float32')
            max_char_len = character_dim*max_word_len
            # samples_char = np.zeros([samples.shape[0], max_char_len], dtype='int32')

            for word, idx in word_index.items():
                for k,c in enumerate(word):
                    if k >= max_word_len: break
                    if not c in embedvec: continue
                    embedmat_char[idx, k] = embedvec[c]
            #embedmat = np.concatenate([embedmat, embedmat_char], axis=1)
        data = (samples, labels, labels_index, word_index, (embedmat, embedmat_char), nbs)
        print("dump data to", pkfile)
        pk.dump(data, open(pkfile, "wb"))
    print("\nsamples:", data[0].shape)
    print("labels:", data[1].shape)
    print("#classes:", len(data[2]))
    print("vocab:", len(data[3]))
    print("embeding matrix:", data[4][0].shape)
    if character_dim>0: print("character embeding matrix:", data[4][1].shape)
    #print("wordvect2:", data[4].shape)
    return data

def clean_str(string, lower=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r" +", " ", string)
    return string.strip() if not lower else string.strip().lower()

if __name__ == "__main__":
    process(embed='google', regen=True)
    #process(pkfile='datasets/data_glove.p', embed='glove', regen=True)
