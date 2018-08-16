# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-08-16 15:18:52
'''


class BatchManager(object):
    
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.data = []
        self.batch_data = []

        self.head_vocab = {"unk":0}
        self.tail_vocab = {"unk":0}
        self.relation_vocab = {"unk":0}

        self.load_data()
        self.prepare_batch()

    def add_vocab(self, word, vocab={}):
        if word not in vocab:
            vocab[word] = len(vocab.keys())
        return vocab[word]

    def load_data(self):
        with open("data/train") as fopen:
            lines = fopen.readlines()
            for line in lines:
                head, tail, relation = line.strip().split(",")

                h_v = self.add_vocab(head, self.head_vocab)
                t_v = self.add_vocab(tail, self.tail_vocab)
                r_v = self.add_vocab(relation, self.relation_vocab)
                self.data.append([[h_v], [t_v], [r_v]])
            
            self.head_vocab_size = len(self.head_vocab) + 1
            self.tail_vocab_size = len(self.tail_vocab) + 1
            self.relation_vocab_size = len(self.relation_vocab) + 1

    def prepare_batch(self):
        index = 0
        while True:
            if index + self.batch_size >= len(self.data):
                data = self.data[-self.batch_size:]
                self.batch_data.append(data)
                break
            else:
                data = self.data[index:index+self.batch_size]
                index += self.batch_size
                self.batch_data.append(data)
    
    def iteration(self):
        idx = 0 
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data