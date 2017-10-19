
##  导读  ##
本文讨论了最新爆款论文(Training RNNs as Fast as CNNs)提出的LSTM变种SRU(Simple Recurrent Unit)，以及基于pytorch实现了SRU,并且在四个句子分类的数据集上测试了准确性以及与LSTM、CNN的速度对比。

##  一 、为什么要提出SRU？ ##
深度学习的许多进展目前很多均是来源于增加的模型能力以及相关的计算，这经常涉及到更大、更深的深层神经网络，然而，虽然深层神经网络带来了明显的提升，但是也耗费了巨大的训练时间，特别是在语音识别以及机器翻译的模型训练上，要想获得一个最优的模型，往往要耗费几天的时间。  
为了解决训练模型的计算能力，像利用GPU进行加速训练一样的并行化方法在深度学习领域已经广泛使用，使用GPU进行加速训练的CNN速度提升的异常明显，但是，像RNN、LSTM却无法实现并行化方法，熟悉RNN、LSTM的人都知道，在其典型的实现中，要想计算 h<sub>t</sub> 必须等到前一时刻h<sub>t-1</sub>计算完成，这明显的限制了其实现并行化处理，然而论文提出的简单循环单元(SRU)解除了这种限制，h<sub>t</sub> 的计算不在依赖于前一时刻的计算，这就可以实现并行化处理，训练速度要比LSTM快的多，能够达到想媲美CNN的训练速度。

##  二 、SRU实现及其优化  ##

### 1、SRU实现  ###
熟悉LSTM和GRU的人都知道，它们是根据神经门来控制信息流来缓解梯度消失与梯度爆炸问题，所以，接下来我们看一下典型的SRU实现。  
我们首先对输入的x进行简单的线性变换：  
![](https://i.imgur.com/FO7vJIB.jpg)  
接下来结算遗忘门（forget gate）和 输入门，他们两个都是Sigmoid门：  
![](https://i.imgur.com/3XfUUus.jpg)  
接下来我们计算c，在计算c的过程中，我们使用了共轭表达式 i<sub>t</sub> =  1 - f<sub>t</sub> 来简化运算:    
![](https://i.imgur.com/I7XCySI.jpg)  
最后，我们把c传递给激活函数来计算最终的输出h:  
![](https://i.imgur.com/h86Ytp9.jpg)  
以上就是SRU的几个经典实现，熟悉LSTM的人一定能够看出来，SRU与LSTM一样都是依赖于前一时刻的计算，这样的做法没有什么意义，接下来我们我们在对其进一步的改进。  

**论文的实现中用到了两个格外的特征：**  

- **Skip Connection**  
具体来说，skip connection就是Highway Connection，对训练深层神经网络很有效果，我们来具体看一下公式：  
先设置一个 reset gate，和遗忘门、输入门一样都是Sigmoid门：  
![](https://i.imgur.com/5ygUmu1.jpg)  
然后利用Skip connection，h<sub>t</sub><sup>'</sup> 就是最后的输出：  
![](https://i.imgur.com/3RGY8p8.jpg)  
在后文的测试中，为什单层的SRU很难达到与LSTM相同的效果，而堆叠起来的多层SRU能够达到与LSTM相差无几甚至更好的效果，这里起到了很大的作用。  


-  **Variational dropout**  
为了RNN的正则化除了使用标准的dropout外，还使用了Variational dropout，Variational dropout 在不同的时间步骤 t 上共享 dropout mask。在 RNN 每一个矩阵乘法计算中（即 W * drop(x)），mask 需要应用到输入 x。标准的 dropout 是在 h上执行的，即没有馈送到高速连接的输出状态。

### 2、SRU加速优化  ###
根据上文中的公式看出 f<sub>t</sub> 、 r<sub>t</sub> 都与  h<sub>t-1</sub>  有关，也就是要想计算 h<sub>t</sub> 必须等到前一时刻h<sub>t-1</sub>计算完成，这样就破换了并行性和独立性，无法实现并行化处理，针对此问题，提出了完全drop连接，就是去除了 h<sub>t-1</sub> 的依赖，以下是SRU的公式：  
![](https://i.imgur.com/PxtjnCx.jpg)  
从上述（8）、（9）、（10）三个公式中可以看出，已经解除了h<sub>t-1</sub> 的依赖，这样依赖就可以实现程序的并行化处理，而公式（11），（12）能够非常迅速和简洁的执行计算，因为它们的运算都是对应元素之间的操作。


### 3、CUDA优化  ###
在上述公式8 --- 10中，虽然解除了前一时刻的依赖，但是仍然存在一定的瓶颈，就是三个矩阵乘法的运算，在这里提供了更深的优化策略。  

- 矩阵乘法在所有的时间步骤中可以进行批处理，可以显著的提高计算的强度和提高GPU的利用率，在8 --- 10 的公式中，可以把矩阵乘法可以合成一个，以后的处理就可以根据索引查找，具体如下：  
![](https://i.imgur.com/iPKYi1T.jpg)  

- 对于序列中的元素间的操作可以编译合并到一个内核函数中并在隐藏维度上并行化。
	






## References  ##
[1] Tao Lei and Yu Zhang. Training RNNs as Fast as CNNs. arXiv:1709.02755, 2017.

[2] Tao Lei and Yu Zhang. Training RNNs as Fast as CNNs. arXiv:1709.02755v2, 2017.

[3] James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. Quasi-recurrent neural
networks. In ICLR, 2017.

[4] Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent
neural networks. In Advances in Neural Information Processing Systems 29 (NIPS), 2016.

[5] Jeremy Appleyard, Tomas Kocisky, and Phil Blunsom. Optimizing performance of recurrent neural networks on gpus. arXiv preprint arXiv:1604.01946, 2016.

