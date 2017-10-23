## Introduction
SRU implement in pytorch（Training RNNs as Fast as CNNs） [https://arxiv.org/abs/1709.02755](https://arxiv.org/abs/1709.02755)

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1（if version 0.2.0 is failed, try the 0.1.0）
* numpy

## performance Test Result
- The following is the acc in the four datasets(CR、MR、Subj、Twitter)
- ![](https://i.imgur.com/raMPaTW.jpg)

## Speed Test Result
- The following is the test of speed among CNN、LSTM、SRU（author）
- ![](https://i.imgur.com/vWhHe3G.jpg)
 
- ![](https://i.imgur.com/IZkPNiE.jpg)

## How to use the folder or file

- the file of **hyperparams.py** contains all hyperparams that need to modify, based on yours nedds, select neural networks what you want and config the hyperparams.

- the file of **main-hyperparams.py** is the main function,run the command ("python main_hyperparams.py") to execute the demo.

- the file of **main-hyperparams-CV.py** is the main function for CV data like MR/CR.

- the folder of **models** contains all neural networks models.

- the file of **train_ALL_CNN.py** is the train function about CNN

- the file of **train_ALL_LSTM.py** is the train function about LSTM

- the file of **train_ALL_SRU.py** is the train function about SRU

- the folder of **loaddata** contains some file of load dataset

- the folder of **word2vec** is the file of word embedding that you want to use

- the folder of **data** contains the dataset file,contains train data,dev data,test data.

- the folder of **Temp_Test_Result** save the temp result for cv function.

- the file of **temp_train.txt** is being used to save tarin data fot cv.

- the file of **temp_test.txt.txt** is being used to save test data fot cv..

- the file of **Parameters.txt** is being used to save all parameters values.

- the file of **Test_Result.txt** is being used to save the result of test,in the demo,save a model and test a model immediately,and int the end of training, will calculate the best result value.

## How to use the Word Embedding in demo? 

- the word embedding file saved in the folder of **word2vec**, but now is empty, because of it is to big,so if you want to use word embedding,you can to download word2vec or glove file, then saved in the folder of word2vec,and make the option of word_Embedding to True and modifiy the value of word_Embedding_Path in the **hyperparams.py** file.


## SRU  Networks

1. **model_BiLSTM_1.py** is a simple bidirection LSTM neural networks model.

2. **model_CNN.py** is a simple CNN neural networks model.

3. **model_SRU.py**  and  **model_BiSRU.py**  is a simple  SRU  neural networks model that use the source code of **cuda_functional.py**(the SRU author code, [https://github.com/taolei87/sru](https://github.com/taolei87/sru)).

4. **model_SRU_Formula.py** is a simple SRU neural networks model that use the SRU formula.


## How to config hyperparams in the file of hyperparams.py

- **learning_rate**: initial learning rate.

- **learning_rate_decay**: change the learning rate for optim.

- **epochs**:number of epochs for train

- **batch_size**：batch size for training

- **log_interval**：how many steps to wait before logging training status

- **test_interval**：how many steps to wait before testing

- **save_interval**：how many steps to wait before saving

- **save_dir**：where to save the snapshot

- **Twitter_path、MR_path、CR_path、Subj_path**：datafile path

- **Twitter、MR、CR、Subj**：change the dataset to load

- **shuffle**:whether to shuffle the dataset when load dataset

- **epochs_shuffle**:whether to shuffle the dataset when train in every epoch

- **TWO-CLASS-TASK**:execute two-classification-task 

- **dropout**:the probability for dropout

- **dropout_embed**:the probability for dropout

- **max_norm**:l2 constraint of parameters

- **clip-max-norm**:the values of prevent the explosion and Vanishing in Gradient

- **kernel_sizes**:comma-separated kernel size to use for convolution

- **kernel_num**:number of each kind of kernel

- **static**:whether to update the gradient during train

- **Adam**:select the optimizer of adam

- **SGD**：select the optimizer of SGD

- **Adadelta**:select the optimizer of Adadelta

- **optim-momentum-value**:the parameter in the optimizer

- **wide_conv**:whether to use wide convcolution True : wide  False : narrow

- **min_freq**:min freq to include during built the vocab when use torchtext, default is 1

- **word_Embedding**: use word embedding

- **fix_Embedding**: use word embedding if to fix during trainging

- **embed_dim**:number of embedding dimension

- **word-Embedding-Path**:the path of word embedding file

- **lstm-hidden-dim**:the hidden dim with lstm model

- **lstm-num-layers**:the num of hidden layers with lstm

- **no_cuda**:  use cuda

- **num_threads**:set the value of threads when run the demo

- **init_weight**:whether to init weight

- **init-weight-value**:the value of init weight

- **weight-decay**:L2 weight_decay,default value is zero in optimizer

- **seed_num**:set the num of random seed

- **rm_model**:whether to delete the model after test acc so that to save space


## Reference 

[1] Tao Lei and Yu Zhang. Training RNNs as Fast as CNNs. arXiv:1709.02755, 2017.  
[2] James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. Quasi-recurrent neural
networks. In ICLR, 2017.  
[3] Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent
neural networks. In Advances in Neural Information Processing Systems 29 (NIPS), 2016.  
[4] Jeremy Appleyard, Tomas Kocisky, and Phil Blunsom. Optimizing performance of recurrent neural networks on gpus. arXiv preprint arXiv:1604.01946, 2016.  

