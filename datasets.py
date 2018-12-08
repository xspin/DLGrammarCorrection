from util.data import Dataset
import util.preprocess as upp
import os, sys
import numpy as np
import random


class Lang8v1(Dataset):
    def __init__(self, embedding_method='glove'):
        Dataset.__init__(self)
        self.max_text_length = 40
        self.min_text_length = 3
        self.embedding_method = embedding_method
        self.character_dim = None
        self.name = 'lang8v1'
        self.src_vocab_size = 30000
        self.tgt_vocab_size = 30000

    def load_data(self):
        train_data_path = 'dataset/lang-8-en-1.0/entries.train'
        test_data_path = 'dataset/lang-8-en-1.0/entries.test'
        num = {'train': 1136634, 'test': 10815}
        train_src, train_tgt = self.preprocess(train_data_path, num['train'])
        test_src, test_tgt = self.preprocess(test_data_path, num['test'])
        self.nb_train = len(train_src)
        self.nb_test = len(test_src)
        train_src.extend(test_src)
        train_tgt.extend(test_tgt)
        return train_src, train_tgt, self.src_vocab_size, self.tgt_vocab_size, (self.nb_train, self.nb_test, self.src_vocab_size, self.tgt_vocab_size )

    def preprocess(self, data_path, num=None):
        '''loading and tokenizing the texts
        '''
        print('  preprocessing', data_path)
        cnt = 0
        sources, targets = [], []
        with open(data_path, 'r') as f:
            for i,line in enumerate(f):
                # if i%10000 == 0: print('  processing line', i)
                if num and i%int(num//20)==0: 
                    print('    {:.2f}%'.format(i*100/num)); 
                    # if i>0: break
                col = line.split('\t')
                if len(col) < 5: continue
                x_txt = col[4]
                y_txt = col[5] if len(col) > 5 else x_txt
                x_sent_token = upp.text_process(x_txt)
                y_sent_token = upp.text_process(y_txt)
                if len(x_sent_token) != len(y_sent_token): 
                    cnt += 1
                    continue
                for xtokens, ytokens in zip(x_sent_token, y_sent_token):
                    if len(xtokens) > self.max_text_length or len(xtokens)<self.min_text_length: continue
                    sources.append(xtokens)
                    targets.append(ytokens)
                    temp = upp.add_noise(xtokens)
                    if temp:
                        sources.append(temp)
                        targets.append(ytokens)
        print('  ignored:', cnt)
        return sources, targets


if __name__ == '__main__':
    data = Lang8v1()
    data.process(regen=True)
    data.show()