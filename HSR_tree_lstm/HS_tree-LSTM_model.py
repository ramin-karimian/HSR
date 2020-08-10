import data_utils as utils
import numpy as np
import tensorflow as tf
import random
import tree_structured_lstm
import pdb
import os
import time
import pickle

package=""
version=""
model_version="HS_tree-lstm-model_V01"

num_for_hs=30
data_dir=f"HSR_tree_lstm\\pickles"
filename=f"words-hashtags_dictionary_{package}_hastags{num_for_hs}.pkl"

class Config(object):
    num_emb=None
    emb_dim = 300
    hidden_dim = 150
    output_dim=None
    degree = 2
    num_epochs = 10
    early_stopping = 10
    dropout = 0.5
    lr = 0.05
    emb_lr = 0.1
    reg=0.0001
    batch_size = 1 ##important
    maxseqlen = None
    maxnodesize = None
    trainable_embeddings=False
    embedding_path= f"pickles\\emd_matrix-{package}_{version}.pkl"
    embedding=None

def train(restore=False):
    config=Config()
    data,vocab = utils.load_HS_treebank(data_dir,filename)
    config.embedding = utils.load_embeding_matrix(config.embedding_path)
    print( "config.embedding ", np.shape(config.embedding) )
    print( " len vocab.word2idx ", len(vocab.word2idx) )
    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print (' train : ', len(train_set),' dev : ', len(dev_set),' test : ', len(test_set))
    train.train_set=train_set
    num_emb = len(vocab)
    num_labels=len(train_set[0][1])
    print ('num emb: ', num_emb,' num labels: ', num_labels)
    config.num_emb=num_emb
    config.output_dim = num_labels
    config.num_of_hs = num_labels
    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)
    print (config.maxnodesize,config.maxseqlen ," maxsize")
    random.seed()
    np.random.seed()
    with tf.Graph().as_default():
        model = tree_structured_lstm.tf_NarytreeLSTM(config)
        init=tf.global_variables_initializer()
        saver=tf.compat.v1.train.Saver()
        best_valid_score=0.0
        best_valid_epoch=0

        with tf.Session() as sess:
            sess.run(init)
            if restore:
                saver.restore(sess,f'./ckpt/tree_lstm/{model_version}/tree_rnn_weights')
                with open(f"ckpt/tree_lstm/{model_version}/results({model_version}).pkl","rb") as f:
                    temp=pickle.load(f)
                    best_valid_score=temp[-2]["acc"]
            hist=[]
            for epoch in range(config.num_epochs):
                start_time=time.time()
                print ('epoch', epoch)
                avg_loss = train_epoch(model, train_set,sess)
                print ('avg loss:', avg_loss, " epoch : ", epoch)
                acc, mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1 \
                    =evaluate(model,dev_set,sess)
                print(f"EVAL:\n acc: {acc}\n mic_f1 : {mic_f1}\n  macro_percision: {macro_percision}\n  macro_recal: {macro_recal}\n  macro_f1: {macro_f1}\n weighted_percision: {weighted_percision}\n weighted_recal: {weighted_recal}\n weighted_f1: {weighted_f1}")


                hist.append({f"EVAL epoch":epoch,f"acc":acc,f"mic_f1":mic_f1,
                              f"macro_percision":macro_percision, f"macro_recal":macro_recal,
                              f"macro_f1":macro_f1 ,f"weighted_percision":weighted_percision,
                              f"weighted_recal":weighted_recal, "weighted_f1":weighted_f1,
                              f"train avg_loss ":avg_loss})
                if acc > best_valid_score:
                    best_valid_score=acc
                    best_valid_epoch=epoch

                    saver.save(sess,f'./ckpt/tree_lstm/{model_version}/tree_rnn_weights')
                if epoch -best_valid_epoch > config.early_stopping:
                    break
                print ("time per epochis {0}".format(time.time()-start_time))

            acc, mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1\
                = evaluate(model,test_set,sess)
            print(f"TEST :\n acc: {acc}\n mic_f1 : {mic_f1}\n  macro_percision: {macro_percision}\n macro_recal: {macro_recal}\n macro_f1: {macro_f1}\n weighted_percision: {weighted_percision}\n weighted_recal: {weighted_recal}\n weighted_f1: {weighted_f1}")

            hist.append({f"TEST epoch":epoch,f"acc":acc,f"mic_f1" :mic_f1,
              f"macro_percision":macro_percision, f"macro_recal":macro_recal,
              f"macro_f1":macro_f1 ,f"weighted_percision":weighted_percision,
              f"weighted_recal":weighted_recal, "weighted_f1":weighted_f1})
            return hist

def train_epoch(model,data,sess):
    loss=model.train(data,sess)
    return loss

def evaluate(model,data,sess):
    acc,mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1\
        = model.evaluate(data,sess)
    return acc,mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1

if __name__ == '__main__':

    train(False)

