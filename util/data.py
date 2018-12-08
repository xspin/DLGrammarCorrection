import numpy as np
import util.preprocess as upp 
from util.config import *

class Dataset:
    def __init__(self):
        self.embedding_method = ''
        self.character_dim = None
        self.name = ''
        self.nb_train = 0
        self.nb_test = 0

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.x_dev = []
        self.y_dev = []
        self.src_embedding_matrix = []
        self.tgt_embedding_matrix = []
        self.embedding_matrix_char = []
        self.src_word_index = []
        self.tgt_word_index = []
        #self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    def show(self):
        print('-'*34)
        # print('x_train:', self.x_train.shape)
        # print('y_train:', self.y_train.shape)
        # print('x_test:', self.x_test.shape)
        # print('y_test:', self.y_test.shape)
        # print('x_dev:', self.x_dev.shape)
        # print('y_dev:', self.y_dev.shape)
        print('train src: max/min/mean/std: {:.0f}/{:.0f}/{:.2f}/{:.2f}'.format(*upp.stat(self.x_train)))
        print('train tgt: max/min/mean/std: {:.0f}/{:.0f}/{:.2f}/{:.2f}'.format(*upp.stat(self.y_train)))
        print('train: {}, dev: {}, test: {}'.format(self.nb_train, self.nb_dev, self.nb_test))
        print('src embedding matrix: {}  {:.2f} MB'.format(self.src_embedding_matrix.shape, self.src_embedding_matrix.nbytes/(1<<20)))
        print('tgt embedding matrix: {} {:.2f} MB'.format(self.tgt_embedding_matrix.shape, self.tgt_embedding_matrix.nbytes/(1<<20)))
        print('Embedding method:', self.embedding_method)
        print('-'*34)

    def load_data(self):
        '''Loading data from the original files. It is required to be implemented for each dataset

        Args: 
            self
        Returns:
            texts, labels, word_index, max_text_length, (self.nb_train, self.nb_test)
        '''
        pass

    def process(self, regen=False):
        '''Process the dataset

        Args:
            regen: a boolean flag indicating if re-generating the data
        '''
        temp = upp.process(self.name, self.load_data, embed=self.embedding_method, regen=regen, character_dim = self.character_dim)
        # data, labels, labels_index, word_index, embedding_matrix, nbs = temp
        (src_sequences, tgt_sequences, src_word_index, tgt_word_index, src_embedmat, tgt_embedmat, nbs) = temp
        self.nb_train, self.nb_test, self.src_vocab_size, self.tgt_vocab_size = nbs
        self.x_train = src_sequences[:self.nb_train]
        self.y_train = tgt_sequences[:self.nb_train]
        self.x_test = src_sequences[self.nb_train:self.nb_train+self.nb_test]
        self.y_test = tgt_sequences[self.nb_train:self.nb_train+self.nb_test]
        self.x_dev = src_sequences[self.nb_train+self.nb_test:]
        self.y_dev = tgt_sequences[self.nb_train+self.nb_test:]
        self.nb_dev = len(self.x_dev)
        self.src_word_index = src_word_index
        self.tgt_word_index = tgt_word_index
        self.src_embedding_matrix = src_embedmat
        self.tgt_embedding_matrix = tgt_embedmat
        self.get_data()
        self.src_i2w = {i:w for w,i in self.src_word_index.items()}
        self.tgt_i2w = {i:w for w,i in self.tgt_word_index.items()}
        for w,i in SVOCAB.items(): 
            self.src_i2w[i] = w
            self.tgt_i2w[i] = w

    def get_data(self):
        if len(self.x_dev)==0: 
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            k = int(len(indices)*0.9)
            self.x_train = np.asarray(self.x_train)
            self.y_train = np.asarray(self.y_train)
            self.x_test = np.asarray(self.x_test)
            self.y_test = np.asarray(self.y_test)
            self.x_dev = self.x_train[indices[k:]]
            self.y_dev = self.y_train[indices[k:]]
            self.x_train = self.x_train[indices[:k]]
            self.y_train = self.y_train[indices[:k]]
            self.nb_dev = len(self.x_dev)
        return self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test

    def get_batch(self, batch_size, flag='train', shuffle=True):
        if flag == 'train':
            sources, targets = self.x_train, self.y_train
        elif flag == 'test':
            sources, targets = self.x_test, self.y_test
        elif flag == 'dev':
            sources, targets = self.x_dev, self.y_dev
        else:
            assert ValueError('invalid flag: {}'.format(flag))
        if shuffle:
            indices = np.arange(len(sources))
            np.random.shuffle(indices)
            sources = sources[indices]
            targets = targets[indices]
        idx, next_idx = 0, 0
        while idx < len(sources):
            next_idx = idx+batch_size
            if next_idx > len(sources): break
            src_len = np.asarray(list(map(len, sources[idx:next_idx])), dtype='int32')
            tgt_len = np.asarray(list(map(len, targets[idx:next_idx])), dtype='int32')
            max_src_len = np.max(src_len)
            max_tgt_len = np.max(tgt_len)
            src_batch = np.zeros([next_idx-idx, max_src_len], dtype='int32')
            tgt_batch = np.zeros([next_idx-idx, max_tgt_len+1], dtype='int32')
            for i in range(idx, next_idx):
                src_batch[i-idx] = sources[i]+[SVOCAB[PAD]]*(max_src_len-len(sources[i]))
                tgt_batch[i-idx] = targets[i]+[SVOCAB[PAD]]*(max_tgt_len-len(targets[i])) + [SVOCAB[EOS]]
            yield np.asarray(src_batch, dtype=np.int32), np.asarray(tgt_batch, dtype=np.int32), src_len, tgt_len+1
            idx = next_idx

