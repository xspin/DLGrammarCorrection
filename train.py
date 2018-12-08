import tensorflow as tf
import numpy as np
import random
import time
from model import Seq2seq
import datasets
import os
from util.config import *
import logging

# logging.basicConfig(level=logging.INFO)

logFormatter = logging.Formatter("%(asctime)s %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("log.txt")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.level = logging.INFO

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf_config = tf.ConfigProto(allow_soft_placement=True)
# tf_config.gpu_options.allow_growth = True 

def sec2str(sec):
  h = sec//3600
  sec = sec%3600
  m = sec//60
  sec = sec%60
  return "{:.0f}h{:02.0f}m{:02.0f}s".format(h,m,sec)

def findLatestCheckpointBatch(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    files = list(filter(lambda x:x[-5:]=='index', files))
    max_epoch = 0
    for fname in files:
        lidx = fname.find('-')
        ridx = fname.rfind('.')
        s = fname[lidx:ridx]
        if s=='': continue
        max_epoch = max(-int(s), max_epoch)
    return max_epoch


class Config(object):
    # embedding_dim = 100
    batch_size = 64
    epochs = 100
    learning_rate = 0.005
    source_vocab_size = None
    target_vocab_size = None
    useTeacherForcing = True
    useAttention = True
    encoder_dims = [64, 64]
    hidden_dim = encoder_dims[-1]
    keep_prob = 0.9
    print_every = 100
    save_every = 2000
    checkpoint_dir = 'checkpoint'

debug = True

def main():
    logging.info("(1) load data......")
    data = datasets.Lang8v1()
    data.process()
    data.show()
    # docs_source, docs_target = load_data("")
    # w2i_source, i2w_source = make_vocab(docs_source)
    # w2i_target, i2w_target = make_vocab(docs_target)

    config = Config()
    config.source_vocab_size = data.src_vocab_size
    config.target_vocab_size = data.tgt_vocab_size

    logging.info("(2) build model......")
    model = Seq2seq(config=config, 
                    src_embedding=data.src_embedding_matrix, 
                    tgt_embedding=data.tgt_embedding_matrix,  
                    useTeacherForcing=config.useTeacherForcing, 
                    useAttention=config.useAttention)
    
    logging.info("(3) run model......")
    with tf.Session(config=tf_config) as sess:
        tf.summary.FileWriter('graph', sess.graph)
        model.init(sess)
        best_epoch = 0
        previous_losses = []
        exp_loss = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        batches_per_epoch = data.nb_train/config.batch_size
        time_per_iter = None

        checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)
        # last_epoch = -int(checkpoint_path[checkpoint_path.rfind('-'):])
        last_epoch = findLatestCheckpointBatch(config.checkpoint_dir)
        # checkpoint_path = os.path.join('checkpoint', "best.ckpt-2")
        logging.info('last epoch: %s'%last_epoch)
        if debug: exit()
        if os.path.exists('checkpoint/checkpoint'): 
            logging.info('Restore model from %s'% checkpoint_path)
            model.saver.restore(sess, checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            exit()
        if debug: exit()
        for epoch in range(last_epoch+1, config.epochs):
            epoch_tic = time.time()
            current_step = 0
            # for source_tokens, source_mask, target_tokens, target_mask in pair_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_layers):
            for batch_vars in data.get_batch(config.batch_size, 'train'):
                src_batch, tgt_batch, src_lens, tgt_lens = batch_vars
                # Get a batch and make a step.
                tic = time.time()
                loss, grad_norm, param_norm = model.train(*batch_vars)
                toc = time.time()
                iter_time = toc - tic
                # total_iters += np.sum(target_mask)
                # tps = total_iters / (time.time() - start_time)
                current_step += 1
                # if current_step>5: break
                time_per_iter = (time.time()-epoch_tic)/current_step

                # lengths = np.sum(target_mask, axis=0)
                mean_length = np.mean(src_lens)
                std_length = np.std(src_lens)

                if not exp_loss:
                    exp_loss = loss
                    exp_length = mean_length
                    exp_norm = grad_norm
                else:
                    exp_loss = 0.99*exp_loss + 0.01*loss
                    exp_length = 0.99*exp_length + 0.01*mean_length
                    exp_norm = 0.99*exp_norm + 0.01*grad_norm

                loss = loss / mean_length

                if current_step==1 or current_step % config.print_every == 0:
                    logging.info(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    logging.info('epoch %d/%d, batch %d/%.0f\n  loss %f, exp_loss %f, grad norm %f, param norm %f, length mean/std %f/%f' %
                            (epoch, config.epochs, current_step, batches_per_epoch, loss, exp_loss / exp_length, grad_norm, param_norm, mean_length, std_length))
                    logging.info('Cost Time: {}, ETA: {}, iter time: {:.3f} sec\n'.format(sec2str(toc-start_time), 
                                    sec2str(time_per_iter*(batches_per_epoch-current_step)), 
                                    (time_per_iter)))

                    predict_batch = model.predict(*batch_vars)
                    logging.info('-'*80)
                    for i in range(3):
                        logging.info("[src]: "+ ' '.join([data.src_i2w[num] for num in src_batch[i] if data.src_i2w[num] != PAD]))
                        logging.info("[tgt]: "+ ' '.join([data.tgt_i2w[num] for num in tgt_batch[i] if data.tgt_i2w[num] != PAD]))
                        logging.info("[prd]: "+ ' '.join([data.tgt_i2w[num] for num in predict_batch[i] if data.tgt_i2w[num] != PAD]))
                        logging.info('-')
                    logging.info('-'*80)
                    logging.info("")

                if current_step % config.save_every == 0:
                    logging.info('Saving model to {}'.format(checkpoint_path))
                    model.saver.save(sess, checkpoint_path)


            epoch_toc = time.time()
            logging.info('Cost Time: {}, Total ETA: {}\n'.format(sec2str(epoch_toc-start_time), 
                    sec2str((epoch_toc-epoch_tic)*(config.epochs-epoch))))

            ## Validate
            # valid_cost = validate(model, sess, x_dev, y_dev)
            logging.info('validation ...')
            loss_dev =[] 
            tot_iter = data.nb_dev/config.batch_size
            # nb_dev = config.batch_size*tot_iter
            for i, dev_batch in enumerate(data.get_batch(config.batch_size, 'dev')):
                t = model.test(*dev_batch)
                loss_dev.append(t)
                if i % max(1,tot_iter//20) == 0:
                    logging.info('  {:.2f}%  loss: {:.2f}'.format((i+1)*100/tot_iter, t))
                if i+1==tot_iter: break
            valid_loss = np.mean(loss_dev)

            logging.info("Epoch %d Validation cost: %.2f time: %s" % (epoch, valid_loss, sec2str(epoch_toc - epoch_tic)))

            ## Checkpoint
            checkpoint_path = os.path.join(config.checkpoint_dir, "best.ckpt")
            if len(previous_losses) > 2 and valid_loss > previous_losses[-1]:
                pass
                # logging.info("Annealing learning rate by %f" % FLAGS.learning_rate_decay_factor)
                # sess.run(model.learning_rate_decay_op)
                # model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
            # else:
            logging.info('Saving checkpoint to {}'.format(checkpoint_path))
            previous_losses.append(valid_loss)
            # best_epoch = epoch
            model.saver.save(sess, checkpoint_path, global_step=epoch)
            with open('checkpoint/log', 'a') as f:
                f.write('{:02d}: {:.6f}\n'.format(epoch, valid_loss))
    
if __name__ == "__main__":
    main()