import tensorflow as tf
from util.config import *
import numpy as np

class Seq2seq(object):
    
    def build_inputs(self, config):
        with tf.name_scope('Input'):
            self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
            self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
            self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
            self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')
    
    def biGRU_layer(self, inputs, dim, seq_len):
        with tf.variable_scope("gru_cell"):
            fw_cell = tf.nn.rnn_cell.GRUCell(dim)
            bw_cell = tf.nn.rnn_cell.GRUCell(dim)
        ((fw_outputs, bw_outputs), (fw_final_state, bw_final_state)) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, 
                                            inputs=inputs, 
                                            sequence_length=seq_len, 
                                            dtype=tf.float32, time_major=False)
        state = tf.add(fw_final_state, bw_final_state)
        outputs = tf.add(fw_outputs, bw_outputs)
        return outputs, state

    def __init__(self, config, src_embedding, tgt_embedding, useTeacherForcing=True, useAttention=True, useBeamSearch=1):
        self.build_inputs(config)

        with tf.variable_scope("Encoder"):
            # encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]), dtype=tf.float32, name='encoder_embedding')
            encoder_embedding = tf.constant(src_embedding, dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)

            # with tf.variable_scope("gru_cell"):
            #     encoder_fw_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            #     encoder_bw_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            # ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
            #     tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell, cell_bw=encoder_bw_cell, 
            #                                     inputs=encoder_inputs_embedded, 
            #                                     sequence_length=self.seq_inputs_length, 
            #                                     dtype=tf.float32, time_major=False)
            # encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
            # encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)
            inputs = encoder_inputs_embedded
            for i, dim in enumerate(config.encoder_dims[:-1]):
                with tf.variable_scope('layer_%d'%i):
                    outputs, state = self.biGRU_layer(inputs, dim, self.seq_inputs_length)
                    inputs = tf.nn.dropout(outputs, config.keep_prob)
            with tf.variable_scope('layer_out'):
                encoder_outputs, encoder_state = self.biGRU_layer(inputs, config.encoder_dims[-1], self.seq_inputs_length)
        
        with tf.variable_scope("Decoder"):
            # decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]), dtype=tf.float32, name='decoder_embedding')
            decoder_embedding = tf.constant(tgt_embedding, dtype=tf.float32, name='decoder_embedding')
            tokens_go = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_SOS') * SVOCAB[SOS]
            if useTeacherForcing:
                decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
                helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs), self.seq_targets_length)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, SVOCAB[EOS])
            # decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
            # helper1 = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs), self.seq_targets_length)
            # helper2 = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, SVOCAB[EOS])
            # helper = tf.cond(tf.random_uniform(shape=[])<useTeacherForcing, lambda: helper1, lambda: helper2)

            with tf.variable_scope("gru_cell"):
                decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
                if useAttention:
                    if useBeamSearch > 1:
                        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=useBeamSearch)
                        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.seq_inputs_length, multiplier=useBeamSearch)
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim, 
                                                                                    memory=tiled_encoder_outputs, 
                                                                                    memory_sequence_length=tiled_sequence_length)
                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                        tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
                        tiled_decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size*useBeamSearch, dtype=tf.float32)
                        tiled_decoder_initial_state = tiled_decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
                        decoder_initial_state = tiled_decoder_initial_state
                    else:
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim, 
                                                                                    memory=encoder_outputs, 
                                                                                    memory_sequence_length=self.seq_inputs_length)
                        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.seq_inputs_length)
                        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
                        decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
                        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                else:
                    if useBeamSearch > 1:
                        decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
                    else:
                        decoder_initial_state = encoder_state
            
            if useBeamSearch > 1:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, tokens_go, SVOCAB[EOS],  
                                                                decoder_initial_state, beam_width=useBeamSearch, 
                                                                output_layer=tf.layers.Dense(config.target_vocab_size))
            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, 
                output_layer=tf.layers.Dense(config.target_vocab_size))
            
            decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                                                maximum_iterations=tf.reduce_max(self.seq_targets_length))
            
        with tf.name_scope('Output'):
            if useBeamSearch > 1:
                self.out = decoder_outputs.predicted_ids[:,:,0]
            else:	
                decoder_logits = decoder_outputs.rnn_output
                self.out = tf.argmax(decoder_logits, 2)
                
                sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.seq_targets, weights=sequence_mask)
                
                self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
                
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                # clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norm = tf.global_norm(gradients)
                self.param_norm = tf.global_norm(params)

    
    def init(self, sess):
        self.sess = sess
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        train_params = tf.trainable_variables()
        all_params = tf.global_variables()
        num_train_params = 0
        num_all_params = 0
        var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
             for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        print('='*50)
        print('Model size: {:.2f} MB'.format(sum(var_sizes) / (1024 ** 2)))
        for tp in train_params:
            num_train_params += np.product(list(map(int, tp.shape)))
        for tp in all_params:
            num_all_params += np.product(list(map(int, tp.shape)))
        print('Trainable params: {:.0f}  {:.2f} MB'.format(num_train_params, num_train_params*4/(1<<20)))
        print('All params: {:.0f}  {:.2f} MB'.format(num_all_params, num_all_params*4/(1<<20)))
        print('='*50)

    def train(self, src_batch, tgt_batch, src_len, tgt_len):
        feed_dict = {
            self.seq_inputs: src_batch,
            self.seq_inputs_length: src_len,
            self.seq_targets: tgt_batch,
            self.seq_targets_length: tgt_len
        }
        output_feed = [self.loss, self.gradient_norm, self.param_norm, self.train_op]
        outputs = self.sess.run(output_feed, feed_dict)
        return outputs[:3]

    def test(self, src_batch, tgt_batch, src_len, tgt_len):
        feed_dict = {
            self.seq_inputs: src_batch,
            self.seq_inputs_length: src_len,
            self.seq_targets: tgt_batch,
            self.seq_targets_length: tgt_len
        }
        loss = self.sess.run(self.loss, feed_dict)
        return loss

    def predict(self, src_batch, tgt_batch, src_len, tgt_len):
        feed_dict = {
            self.seq_inputs: src_batch,
            self.seq_inputs_length: src_len,
            self.seq_targets: tgt_batch,
            self.seq_targets_length: tgt_len
        }
        predict_batch = self.sess.run(self.out, feed_dict)
        return predict_batch

    def predict(self, src_sent, src_len):
        feed_dict = {
            self.seq_inputs: [src_sent],
            self.seq_inputs_length: [src_len],
            # self.seq_targets: tgt_batch,
            self.seq_targets_length: [50]
        }
        predict_batch = self.sess.run(self.out, feed_dict)
        return predict_batch
        