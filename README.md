### Cluster-based-TransR

Reference [Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)


### TransR Model

    tr = TransR()
    tr.predict_relations("磁器口", "重庆")
    > the relation between 磁器口 and 重庆 is 景点
