
##  导读  ##
本文讨论了最新爆款论文(Training RNNs as Fast as CNNs)提出的LSTM变种SRU(Simple Recurrent Unit)，以及基于pytorch实现了SRU,并且在四个句子分类的数据集上测试了准确性以及与LSTM、CNN的速度对比。

##  一 、为什么要提出SRU？ ##
深度学习的许多进展目前很多均是来源于增加的模型能力以及相关的计算，这经常涉及到更大、更深的深层神经网络，然而，虽然深层神经网络带来了明显的提升，但是也耗费了巨大的训练时间，特别是在语音识别以及机器翻译的模型训练上，要想获得一个最优的模型，往往要耗费几天的时间。  
为了解决训练模型的计算能力，像利用GPU进行加速训练一样的并行化方法在深度学习领域已经广泛使用，使用GPU进行加速训练的CNN速度提升的异常明显，但是，像RNN、LSTM却无法实现并行化方法，熟悉RNN、LSTM的人都知道，在其典型的实现中，要想计算"h_t"必须等到前一时刻计算完成_，这明显的限制了其实现并行化处理，然而论文提出的简单循环单元(SRU)解除了这种限制，h_t的计算不在依赖于前一时刻的计算，这就可以实现并行化处理，训练速度要比LSTM快的多，能够达到想媲美CNN的训练速度。

##  二 、Simple Recurrent Unit(SRU)  ##


## References  ##
[1] Tao Lei and Yu Zhang. Training RNNs as Fast as CNNs. arXiv:1709.02755, 2017.

[2] Tao Lei and Yu Zhang. Training RNNs as Fast as CNNs. arXiv:1709.02755v2, 2017.

[3] James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. Quasi-recurrent neural
networks. In ICLR, 2017.

[4] Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent
neural networks. In Advances in Neural Information Processing Systems 29 (NIPS), 2016.

[5] Jeremy Appleyard, Tomas Kocisky, and Phil Blunsom. Optimizing performance of recurrent neural networks on gpus. arXiv preprint arXiv:1604.01946, 2016.

