# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-08-16 10:43:55
'''

import pickle

import numpy as np
import tensorflow as tf

from data_utils import BatchManager

class TransR(object):
    
    def __init__(self):
        self.size_of_relation = 100
        self.size_of_entity = 100

        self.head_input_size = 10
        self.tail_input_size = 10
        self.relation_input_size = 10

        self.checkpoint_dir = "./models/"
        self.checkpoint_path = "./models/transR.ckpt"

    def init_model(self):
        self.__placeholder()
        self.__head()
        self.__tail()
        self.__relation()
        self.__trans()
        self.__optimizer()

    def __placeholder(self):
        self.head_inputs = tf.placeholder(
            shape=[None, 1],
            dtype=tf.int32,
            name="head"
        )
        self.tail_inputs = tf.placeholder(
            shape=[None, 1],
            dtype=tf.int32,
            name="tail"
        )
        self.relation_inputs = tf.placeholder(
            shape=[None, 1],
            dtype=tf.int32,
            name="relation"
        )

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="dropout"
        )

    def __head(self):
        with tf.variable_scope("head_embedding") as scope:
            embedding_matrix = tf.get_variable(
                name="head_embedding_matrix",
                shape=[self.head_input_size, self.size_of_entity],
                dtype=tf.float32
            )
            head_embedding = tf.nn.embedding_lookup(
                embedding_matrix, self.head_inputs
            )
            self.head = tf.nn.dropout(
                head_embedding, self.dropout
            )

    def __tail(self):
        with tf.variable_scope("tail_embedding") as scope:
            embedding_matrix = tf.get_variable(
                name="tail_embedding_matrix",
                shape=[self.tail_input_size, self.size_of_entity],
                dtype=tf.float32
            )
            tail_embedding = tf.nn.embedding_lookup(
                embedding_matrix, self.tail_inputs
            )
            self.tail = tf.nn.dropout(
                tail_embedding, self.dropout
            )

    def __relation(self):
        with tf.variable_scope("relation_embedding") as scope:
            embedding_matrix = tf.get_variable(
                name="relation_embedding_matrix",
                shape=[self.relation_input_size, self.size_of_relation],
                dtype=tf.float32
            )
            relation_embedding = tf.nn.embedding_lookup(
                embedding_matrix, self.relation_inputs
            )
            self.relation = tf.nn.dropout(
                relation_embedding, self.dropout
            )

    def __trans(self):
        with tf.variable_scope("trans") as scope:
            self.Mr = tf.get_variable(
                name="Mr",
                shape=[self.size_of_entity, self.size_of_relation]
            )
            self.head = tf.reshape(self.head, shape=[-1, self.size_of_entity])
            self.tail = tf.reshape(self.tail, shape=[-1, self.size_of_entity])
            self.relation = tf.reshape(self.relation, shape=[-1, self.size_of_relation])

            self.hr = tf.matmul(self.head, self.Mr)
            self.tr = tf.matmul(self.tail, self.Mr)
            self.r = self.relation

            fr = self.hr + self.r - self.tr
            self.logits = tf.reduce_sum(fr * fr, axis=1)
            self.loss = tf.reduce_sum(self.logits)
    
    def __optimizer(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                        gradients, 5)
        # Optimization
        optimizer = tf.train.GradientDescentOptimizer(0.02)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, batch, sess):
        heads = [i[0] for i in batch]
        tails = [i[1] for i in batch]
        relations = [i[2] for i in batch]

        feed = {
            self.head_inputs:heads,
            self.tail_inputs:tails,
            self.relation_inputs:relations,
            self.dropout:0.5
        }
        loss,_ = sess.run([self.loss, self.train_op], feed_dict=feed)
        return loss

    def train(self):
        batch_manager = BatchManager()
        self.head_input_size = batch_manager.head_vocab_size
        self.tail_input_size = batch_manager.tail_vocab_size
        self.relation_input_size = batch_manager.relation_vocab_size
        data_map = {
            "head_size":self.head_input_size,
            "tail_size":self.tail_input_size,
            "relation_size":self.relation_input_size,
            "head_vocab":batch_manager.head_vocab,
            "tail_vocab":batch_manager.tail_vocab,
            "relation_vocab":batch_manager.relation_vocab
        }
        f = open("models/data_map.pkl", "wb")
        pickle.dump(data_map, f)
        f.close()

        self.init_model()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("[->] restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("[->] no model, initializing")
                sess.run(tf.global_variables_initializer())

            for i in range(200):
                print("epoch {}".format(i))
                for batch in batch_manager.get_batch():
                    loss = self.step(batch, sess)
                    print("\tloss: {}".format(loss))
                    self.saver.save(sess, self.checkpoint_path)

    def predict_relations(self, head, tail):
        f = open("models/data_map.pkl", "rb")
        data_map = pickle.load(f)
        f.close()
        
        self.head_vocab = data_map.get("head_vocab")
        self.tail_vocab = data_map.get("tail_vocab")
        self.relation_vocab = data_map.get("relation_vocab")

        self.head_input_size = data_map.get("head_size")
        self.tail_input_size = data_map.get("tail_size")
        self.relation_input_size = data_map.get("relation_size")

        self.init_model()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("[->] restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("[->] no model, initializing")
                sess.run(tf.global_variables_initializer())

            relations = list(self.relation_vocab.keys())
            relations_vec = [[self.relation_vocab.get(i)] for i in relations]

            heads_vec = [[self.head_vocab.get(head, 0)]] * (self.relation_input_size - 1)
            tails_vec = [[self.tail_vocab.get(tail, 0)]] * (self.relation_input_size - 1)

            feed = {
                self.head_inputs:heads_vec,
                self.tail_inputs:tails_vec,
                self.relation_inputs:relations_vec,
                self.dropout:1
            }
            logits = sess.run(self.logits, feed_dict=feed)
            min_index = np.argmin(logits)
            print(logits)
            print("the relation between {} and {} is {}".format(
                head, tail, relations[min_index]
            ))

if __name__ == "__main__":
    tr = TransR()
    tr.predict_relations("北京大学", "北京")
    # tr.train()
