import tensorflow as tf
import numpy as np
import random
import time
from model import Seq2seq
from train import Config
from util.config import *
import datasets
import logging, os
import nltk.tokenize as tok

logging.basicConfig(level=logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

class Corrector():
    def __init__(self):
        logging.info("load data......")
        self.data = datasets.Lang8v1()
        self.data.process()
        self.data.show()

        self.config = Config()
        self.config.source_vocab_size = self.data.src_vocab_size
        self.config.target_vocab_size = self.data.tgt_vocab_size
        self.config.batch_size = 1

        logging.info("build model......")
        self.model = Seq2seq(config=self.config,
                        src_embedding=self.data.src_embedding_matrix, 
                        tgt_embedding=self.data.tgt_embedding_matrix,  
                        useTeacherForcing=False, useAttention=True, useBeamSearch=8)
        
        logging.info("init model......")
        # with tf.Session() as sess:
        sess = tf.Session()
        self.model.init(sess)
        checkpoint_path = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        assert checkpoint_path, 'No checkpoint found'
        logging.info('Restore model from %s'%checkpoint_path)
        self.model.saver.restore(sess, checkpoint_path)

    def correct(self, sentence):
        '''correct a English sentence

        Args:
            sentence: a string of a sentence
        Returns:
            a corrected sentence
        '''
        sentence = sentence.lower()
        sentence = tok.word_tokenize(sentence)
        # print(' '.join(sentence))
        sent = list(map(lambda x:self.data.src_word_index[x] if x in self.data.src_word_index else SVOCAB[UNK], sentence))
        
            # src_batch, tgt_batch, src_lens, tgt_lens = batch_vars
        predict_batch = self.model.predict(sent, len(sent))
        
        print("[src]:", ' '.join([self.data.src_i2w[num] for num in sent if self.data.src_i2w[num] != PAD]))
        # print("[tgt]:", ' '.join([data.tgt_i2w[num] for num in tgt_batch[0] if data.tgt_i2w[num] != PAD]))
        print("[prd]:", ' '.join([self.data.tgt_i2w[num] for num in predict_batch[0] if self.data.tgt_i2w[num] != PAD]))

            
if __name__ == "__main__":
    sentences = [
        "I hates the world!",
        "I like to do the work.",
        "what is you doing?"
    ]
    corrector = Corrector()
    for sent in sentences:
        corrector.correct(sent)
    while True:
        sent = str(input('\nInput: '))
        if sent in ['q', 'exit', 'quit']: break
        corrector.correct(sent)
    print('Bye~')