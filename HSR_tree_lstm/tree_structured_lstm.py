import numpy as np
import tensorflow as tf
import os
import pickle
import sys
from data_utils import  extract_batch_tree_data
from sklearn.metrics import f1_score ,precision_recall_fscore_support

class tf_NarytreeLSTM(object):
    def __init__(self, config):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.output_dim = config.output_dim
        self.config = config
        self.batch_size = config.batch_size
        self.reg = self.config.reg
        self.degree = config.degree
        self.num_of_hs = config.num_of_hs
        self.embedding = config.embedding
        assert self.num_of_hs == self.output_dim
        assert self.emb_dim > 1 and self.hidden_dim > 1

        self.add_placeholders()
        emb_leaves = self.add_embedding()
        self.add_model_variables()
        batch_loss = self.compute_loss(emb_leaves)
        self.loss, self.total_loss = self.calc_batch_loss(batch_loss)
        self.train_op1, self.train_op2 = self.add_training_op()
        # self.train_op=tf.no_op()

    def add_embedding(self):
        with tf.variable_scope("Embed", regularizer=None):
            embedding= tf.get_variable(name="embedding",
                                        shape=[self.num_emb,self.emb_dim],
                                        initializer=tf.constant_initializer(np.array(self.embedding)),
                                        trainable=False,
                                        regularizer=None)               # embedding = tf.get_variable('embedding', [self.num_emb,self.emb_dim], initializer=tf.random_uniform_initializer(-0.05, 0.05),trainable=True,regularizer=None)
            ix = tf.to_int32(tf.not_equal(self.input, -1)) * self.input
            emb_tree = tf.nn.embedding_lookup(embedding, ix)
            emb_tree = emb_tree * (tf.expand_dims(
                tf.to_float(tf.not_equal(self.input, -1)), 2))
            return emb_tree

    def add_placeholders(self):
        dim2 = self.config.maxnodesize
        dim1 = self.config.batch_size
        dim3 = self.num_of_hs
        self.input = tf.placeholder(tf.int32, [dim1, dim2], name='input')
        self.treestr = tf.placeholder(tf.int32, [dim1, dim2, 2], name='tree')
        self.labels = tf.placeholder(tf.int32, [dim1, dim3], name='labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treestr, -1)),[1, 2])  ### it computes total number of nodes for each example(sentence)
        self.n_inodes = self.n_inodes // 2  # it actually gives the real number of the inodes( maxnodesize - inodes)
        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input, -1)), [1])
        self.batch_len = tf.placeholder(tf.int32, name="batch_len")

    def calc_wt_init(self, fan_in=300):
        eps = 1.0 / np.sqrt(fan_in)
        return eps

    def add_model_variables(self):
        with tf.variable_scope("Composition",
                               initializer=
                               tf.contrib.layers.xavier_initializer(),
                               regularizer=
                               tf.contrib.layers.l2_regularizer(self.config.reg )):
            cU = tf.get_variable("cU", [self.emb_dim, 2 * self.hidden_dim],
                                 initializer=tf.random_uniform_initializer(-self.calc_wt_init(), self.calc_wt_init()))
            cW = tf.get_variable("cW", [self.degree * self.hidden_dim, (self.degree + 3) * self.hidden_dim],
                                 initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),
                                                                           self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb", [4 * self.hidden_dim], initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

        with tf.variable_scope("Projection", regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),
                                                                          self.calc_wt_init(self.hidden_dim))
                                )
            bu = tf.get_variable("bu", [self.output_dim], initializer=tf.constant_initializer(0.0)
                                 , regularizer=tf.contrib.layers.l2_regularizer(0.0)
                                 )

    def process_leafs(self, emb):
        with tf.variable_scope("Composition", reuse=True):
            cU = tf.get_variable("cU", [self.emb_dim, 2 * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            b = tf.slice(cb, [0], [2 * self.hidden_dim])
            def _recurseleaf(x):
                concat_uo = tf.matmul(tf.expand_dims(x, 0), cU) + b
                u, o = tf.split(axis=1, num_or_size_splits=2, value=concat_uo)
                o = tf.nn.sigmoid(o)
                u = tf.nn.tanh(u)

                c = u
                h = o * tf.nn.tanh(c)
                hc = tf.concat([h, c], 1)
                hc = tf.squeeze(hc)
                return hc
        hc = tf.map_fn(_recurseleaf, emb)
        return hc

    def compute_loss(self, emb_batch, curr_batch_size=None):
        outloss = []
        prediction = []
        for idx_batch in range(self.config.batch_size):
            tree_states = self.compute_states(emb_batch, idx_batch)
            tree_states = tree_states[-1]  ## in order to get the state of the root node     not this : ##### for the purpose of return_sequence=False  https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
            logits = self.create_output(tree_states)
            labels = tf.gather(self.labels, idx_batch)
            pred = tf.nn.softmax(logits)
            loss = self.calc_loss(pred, labels)

            prediction.append(pred)
            outloss.append(loss)
            print(" logits ", logits,"\n labels ", labels,"\n loss ", loss,"\n pred ", pred)
        batch_loss = tf.stack(outloss)
        self.pred = tf.stack(prediction)
        print(" prediction", prediction,"\n outloss", outloss,"\n batch_loss ", batch_loss,"\n tf.stack(prediction)", tf.stack(prediction))
        return batch_loss

    def compute_states(self, emb, idx_batch=0):
        num_leaves = tf.squeeze(tf.gather(self.num_leaves, idx_batch))
        n_inodes = tf.gather(self.n_inodes, idx_batch)
        embx = tf.gather(tf.gather(emb, idx_batch), tf.range(num_leaves))
        treestr = tf.gather(tf.gather(self.treestr, idx_batch), tf.range(n_inodes))
        leaf_hc = self.process_leafs(embx)
        leaf_h, leaf_c = tf.split(leaf_hc, 2, 1)
        node_h = tf.identity(leaf_h)
        node_c = tf.identity(leaf_c)
        idx_var = tf.constant(0)  # tf.Variable(0,trainable=False)
        with tf.variable_scope("Composition", reuse=True):
            cW = tf.get_variable("cW", [self.degree * self.hidden_dim, (self.degree + 3) * self.hidden_dim])
            cb = tf.get_variable("cb", [4 * self.hidden_dim])
            bu, bo, bi, bf = tf.split(cb, 4, 0)
            def _recurrence(node_h, node_c, idx_var):
                node_info = tf.gather(treestr, idx_var)
                child_h = tf.gather(node_h, node_info)
                child_c = tf.gather(node_c, node_info)
                flat_ = tf.reshape(child_h,[-1])  # sess.run(tf.reshape([[1,2,3],[4,5,6]],[-1]))   aray([1, 2, 3, 4, 5, 6])
                tmp = tf.matmul(tf.expand_dims(flat_, 0), cW)
                u, o, i, fl, fr = tf.split(value=tmp, axis=1, num_or_size_splits=5)
                i = tf.nn.sigmoid(i + bi)
                o = tf.nn.sigmoid(o + bo)
                u = tf.nn.tanh(u + bu)
                fl = tf.nn.sigmoid(fl + bf)
                fr = tf.nn.sigmoid(fr + bf)
                f = tf.concat([fl, fr], 0)
                c = i * u + tf.reduce_sum(f * child_c, [0])
                h = o * tf.nn.tanh(c)
                node_h = tf.concat([node_h, h], 0)
                node_c = tf.concat([node_c, c], 0)
                idx_var = tf.add(idx_var, 1)
                return node_h, node_c, idx_var
            loop_cond = lambda a1, b1, idx_var: tf.less(idx_var, n_inodes)
            loop_vars = [node_h, node_c, idx_var]
            node_h, node_c, idx_var = tf.while_loop(loop_cond, _recurrence,loop_vars, parallel_iterations=10)
            return node_h

    def create_output(self, tree_states):
        with tf.variable_scope("Projection", reuse=True):
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                )
            bu = tf.get_variable("bu", [self.output_dim])

            h = tf.matmul(tf.expand_dims(tree_states, axis=0), U,
                          transpose_b=True) + bu  ##### in order to get the state of the root node
            h = tf.squeeze(h)
            return h

    def calc_loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)         #        l1=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)  ;         #        loss=tf.reduce_sum(l1,[0]) ; return loss
        return loss

    def calc_batch_loss(self, batch_loss):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart = tf.add_n(reg_losses)
        loss = tf.reduce_mean(batch_loss)
        total_loss = loss + 0.5 * regpart
        # total_loss=loss #############################
        return loss, total_loss
        """ for testing purpose ; 
           ba_lo=[[0.5,0.4],[0.3,0.2]]
           reduce_mean(ba_lo).eval()    ==> 0.35000002
           ba_lo1=[[0.4],[0.2]]
           tf.reduce_mean(ba_lo1).eval()  ==> 0.3
           ba_lo2=[0.4,0.2]
           tf.reduce_mean(ba_lo2).eval() ==> 0.3 
        """

    def add_training_op_old(self):
        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(self.total_loss)
        return train_op

    def add_training_op(self):
        loss = self.total_loss
        opt1 = tf.train.AdagradOptimizer(self.config.lr)
        ts = tf.trainable_variables()
        gs = tf.gradients(loss, ts)
        gs_ts = zip(gs, ts)
        gt_emb, gt_nn = [], []
        for g, t in gs_ts:
            print ("t.name,g.name : ",t.name,g.name)
            if "Embed/embedding:0" in t.name:
                gt_emb.append((g, t))
            else:
                gt_nn.append((g, t))
        train_op1 = opt1.apply_gradients(gt_nn)
        train_op = [train_op1, None ]
        return train_op

    def train(self, data, sess):
        data_idxs = range(len(data))
        losses = []
        for i in range(0, len(data), self.batch_size):
            batch_size = min(i + self.batch_size, len(data)) - i
            if batch_size < self.batch_size: break
            batch_idxs = data_idxs[i:i + batch_size]
            batch_data = [data[ix] for ix in batch_idxs]  # [i:i+batch_size]
            input_b, treestr_b, labels_b,_ = extract_batch_tree_data(batch_data, self.num_of_hs, self.config.maxnodesize)
            feed = {self.input: input_b, self.treestr: treestr_b, self.labels: labels_b,
                    self.dropout: self.config.dropout, self.batch_len: len(input_b)}
            loss, _ = sess.run([self.loss, self.train_op1], feed_dict=feed)             # loss, _, _ = sess.run([self.loss, self.train_op1, self.train_op2], feed_dict=feed)
            losses.append(loss)
            avg_loss = np.mean(losses)
            sstr = 'avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(data))
            if i %10000 ==0:
                print(sstr)
            sys.stdout.write(sstr)
            sys.stdout.flush()
        return np.mean(losses)

    def convert_prediction(self,pred,real):
        pred_argmax=np.empty(shape=np.shape(pred),dtype=np.int32)
        for i in range(pred_argmax.shape[0]):
            vec=np.zeros(pred_argmax.shape[1],dtype=np.int32)
            top_k=np.count_nonzero(real[i])
            inds=pred[i].argsort()[-top_k:][::-1]
            vec[inds]=1
            pred_argmax[i]=vec
        return pred_argmax

    def sklearn_f1_score(self,y_pred,y_true):
        y_pred=self.convert_prediction(y_pred,y_true)
        mic_f1 = f1_score(y_true, y_pred, average='micro')
        macro_percision,macro_recal,macro_f1,_ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        weighted_percision,weighted_recal,weighted_f1,_ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1
    def evaluate(self, data, sess):
        num_correct = 0
        total_data = 0
        total_pred = []
        total_label = []
        data_idxs = range(len(data))
        test_batch_size = self.config.batch_size
        for i in range(0, len(data), test_batch_size):
            batch_size = min(i + test_batch_size, len(data)) - i
            if batch_size < test_batch_size: break
            batch_idxs = data_idxs[i:i + batch_size]
            batch_data = [data[ix] for ix in batch_idxs]  # [i:i+batch_size]
            input_b, treestr_b, labels_b,_ = extract_batch_tree_data(batch_data, self.num_of_hs, self.config.maxnodesize)
            feed = {self.input: input_b, self.treestr: treestr_b, self.labels: labels_b,
                    self.dropout: 1.0, self.batch_len: len(input_b)}
            pred_y = sess.run(self.pred, feed_dict=feed)
            total_pred.extend(pred_y)
            total_label.extend(labels_b)
            y = np.argmax(pred_y, axis=1)
            for i in range(labels_b.shape[0]):
                if labels_b[i][y[i]] == 1: num_correct += 1
                total_data += 1
            """ ba_lo=[[0.5,0.4],[0.3,0.2]]  ; ba_lo[0][0]=0   ;  np.argmax(ba_lo,axis=1)   == >  array([1, 0], dtype=int64) """
        acc = float(num_correct) / float(total_data)
        mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1 = self.sklearn_f1_score(total_pred, total_label)
        return acc, mic_f1, macro_percision,macro_recal,macro_f1, weighted_percision,weighted_recal,weighted_f1
