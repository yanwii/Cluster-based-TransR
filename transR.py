# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-08-16 10:43:55
'''

import tensorflow as tf

class TransR(object):
    
    def __init__(self):
        self.size_of_relation = 256
        self.size_of_entity = 128

        self.head_input_size = 10
        self.tail_input_size = 10
        self.relation_input_size = 10

        self.init_model()

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

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            heads = [[1], [3], [5]]
            tails = [[1], [4], [6]]
            relations = [[1], [2], [3]]

            for i in range(200):
                feed = {
                    self.head_inputs:heads,
                    self.tail_inputs:tails,
                    self.relation_inputs:relations,
                    self.dropout:0.5
                }
                loss,_ = sess.run([self.loss, self.train_op], feed_dict=feed)
                print(loss)

            heads = [[1]] * 3
            tails = [[1]] * 3

            feed = {
                self.head_inputs:heads,
                self.tail_inputs:tails,
                self.relation_inputs:[[1], [2], [3]],
                self.dropout:1
            }
            logits = sess.run(self.logits, feed_dict=feed)
            print(logits)
            print(sess.run(tf.argmin(logits)))




tr = TransR()
tr.train()