# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-08-16 10:43:55
'''

import tensorflow as tf

class ClusterBasedTransR(object):
    
    def __init__(self):
        self.size_of_relation = 256
        self.size_of_entity = 128

    def init_model(self):
        pass

    def __trans(self):
        with tf.variable_scope("trans") as scope:
            Mr = tf.get_variable(
                name="Mr",
                shape=[self.size_of_entity, self.size_of_relation]
            )

            hr = tf.matmul(self.h, Mr)
            tr = tf.matmul(self.t, Mr)

        