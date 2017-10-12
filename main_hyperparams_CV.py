#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
from loaddata import loadingdata_Twitter
from loaddata import loadingdata_CV
from loaddata.load_external_word_embedding import Word_Embedding
import train_ALL_CNN
import train_ALL_LSTM
import train_ALL_SRU
from models import model_CNN
from models import model_BiLSTM_1
from models import model_SRU
from models import model_BiSRU
from models import model_SRU_Formula
import multiprocessing as mu
import shutil
import random
import hyperparams
# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

parser = argparse.ArgumentParser(description="sentence classification")
# learning
parser.add_argument('-lr', type=float, default=hyperparams.learning_rate, help='initial learning rate [default: 0.001]')
parser.add_argument('-learning_rate_decay', type=float, default=hyperparams.learning_rate_decay, help='initial learning_rate_decay rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=hyperparams.epochs, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=hyperparams.batch_size, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=hyperparams.log_interval,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=hyperparams.test_interval, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=hyperparams.save_interval, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default=hyperparams.save_dir, help='where to save the snapshot')
# data path
parser.add_argument('-Twitter_path', type=str, default=hyperparams.Twitter_path, help='Twitter data path')
parser.add_argument('-MR_path', type=str, default=hyperparams.MR_path, help='MR data path')
parser.add_argument('-CR_path', type=str, default=hyperparams.CR_path, help='CR data path')
parser.add_argument('-Subj_path', type=str, default=hyperparams.Subj_path, help='Subj data path')
# which data to load
parser.add_argument('-Twitter', action="store_true", default=hyperparams.Twitter, help='load Twitter data')
parser.add_argument('-MR', action="store_true", default=hyperparams.MR, help='load MR data')
parser.add_argument('-CR', action="store_true", default=hyperparams.CR, help='load CR data')
parser.add_argument('-Subj', action="store_true", default=hyperparams.Subj, help='load Subj data')
# shuffle data
parser.add_argument('-shuffle', action='store_true', default=hyperparams.shuffle, help='shuffle the data every epoch' )
parser.add_argument('-epochs_shuffle', action='store_true', default=hyperparams.epochs_shuffle, help='shuffle the data every epoch' )
# use cv
parser.add_argument('-CV', action='store_true', default=hyperparams.CV, help='use cv')
parser.add_argument("-nfold", type=int, default=hyperparams.nfold, help="the cv number")
# task select
parser.add_argument('-TWO_CLASS_TASK', action='store_true', default=hyperparams.TWO_CLASS_TASK, help='whether to execute two-classification-task')
# optim select
parser.add_argument('-Adam', action='store_true', default=hyperparams.Adam, help='whether to select Adam to train')
parser.add_argument('-SGD', action='store_true', default=hyperparams.SGD, help='whether to select SGD to train')
parser.add_argument('-Adadelta', action='store_true', default=hyperparams.Adadelta, help='whether to select Adadelta to train')
# model
parser.add_argument('-rm_model', action='store_true', default=hyperparams.rm_model, help='whether to delete the model after test acc so that to save space')
parser.add_argument('-init_weight', action='store_true', default=hyperparams.init_weight, help='init w')
parser.add_argument('-init_weight_value', type=float, default=hyperparams.init_weight_value, help='value of init w')
parser.add_argument('-init_weight_decay', type=float, default=hyperparams.weight_decay, help='value of init L2 weight_decay')
parser.add_argument('-momentum_value', type=float, default=hyperparams.optim_momentum_value, help='value of momentum in SGD')
parser.add_argument('-init_clip_max_norm', type=float, default=hyperparams.clip_max_norm, help='value of init clip_max_norm')
parser.add_argument('-seed_num', type=float, default=hyperparams.seed_num, help='value of init seed number')
parser.add_argument('-dropout', type=float, default=hyperparams.dropout, help='the probability for dropout [default: 0.5]')
parser.add_argument('-dropout_embed', type=float, default=hyperparams.dropout_embed, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=hyperparams.max_norm, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=hyperparams.embed_dim, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=hyperparams.kernel_num, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default=hyperparams.kernel_sizes, help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=hyperparams.static, help='fix the embedding')
parser.add_argument('-CNN', action='store_true', default=hyperparams.CNN, help='whether to use CNN model')
parser.add_argument('-BiLSTM_1', action='store_true', default=hyperparams.BiLSTM_1, help='whether to use BiLSTM_1 model')
parser.add_argument('-SRU', action='store_true', default=hyperparams.SRU, help='whether to use SRU model')
parser.add_argument('-BiSRU', action='store_true', default=hyperparams.BiSRU, help='whether to use BiSRU model')
parser.add_argument('-SRU_Formula', action='store_true', default=hyperparams.SRU_Formula, help='whether to use SRU_Formula model')
parser.add_argument('-wide_conv', action='store_true', default=hyperparams.wide_conv, help='whether to use wide conv')
parser.add_argument('-fix_Embedding', action='store_true', default=hyperparams.fix_Embedding, help='whether to fix word embedding during training')
parser.add_argument('-word_Embedding', action='store_true', default=hyperparams.word_Embedding, help='whether to load word embedding')
parser.add_argument('-word_Embedding_Path', type=str, default=hyperparams.word_Embedding_Path, help='filename of model snapshot [default: None]')
parser.add_argument('-lstm-hidden-dim', type=int, default=hyperparams.lstm_hidden_dim, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-lstm-num-layers', type=int, default=hyperparams.lstm_num_layers, help='the number of embedding dimension in LSTM hidden layer')
parser.add_argument('-min_freq', type=int, default=hyperparams.min_freq, help='min freq to include during built the vocab')
# nums of threads
parser.add_argument('-num_threads', type=int, default=hyperparams.num_threads, help='the num of threads')
# device
parser.add_argument('-device', type=int, default=hyperparams.device, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no_cuda', action='store_true', default=hyperparams.no_cuda, help='disable the gpu')
# option
args = parser.parse_args()


# load twitter data, no require CV
def load_data_twitter(path,text_field, label_field, **kargs):
    train_data, dev_data, test_data = loadingdata_Twitter.Twitter.splits(path, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    text_field.build_vocab(train_data.text, min_freq=args.min_freq)
    label_field.build_vocab(train_data.label)
    train_iter, dev_iter, test_iter = create_Iterator(train_data, dev_data, test_data, batch_size=args.batch_size,
                                                      **kargs)
    return train_iter, dev_iter, test_iter


# load data that need CV
def load_cv_data(text_field, label_field, path, **kargs):
    train_data, dev_data, test_data = loadingdata_CV.CV.splits(path, text_field, label_field)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(dev_data) {} ".format(len(dev_data)))
    print("len(test_data) {} ".format(len(test_data)))
    text_field.build_vocab(train_data.text, min_freq=args.min_freq)
    label_field.build_vocab(train_data.label)
    train_iter, dev_iter, test_iter = create_Iterator(train_data, dev_data, test_data, batch_size=args.batch_size,
                                                      **kargs)
    return train_iter, dev_iter, test_iter


# create Iterator
def create_Iterator(train_data, dev_data, test_data, batch_size, **kargs):
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(batch_size, len(dev_data), len(test_data)), **kargs)
        # batch_sizes=(batch_size, batch_size, batch_size), **kargs)
    return train_iter, dev_iter, test_iter


# cross CV
# def cv_spilit(data, nfold, test_id):
#     assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
#     print(data.examples[:5])
#     print(data)
#     print("cv")
#     return None, None, None
#
# cross CV
def cv_spilit_file(path, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    print("CV file......")
    if os.path.exists("./temp_train.txt"):
        os.remove("./temp_train.txt")
    if os.path.exists("./temp_test.txt"):
        os.remove("./temp_test.txt")
    file_train = open("./temp_train.txt", "w", encoding="utf-8")
    file_test = open("./temp_test.txt", "w", encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # print(i, line)
            file_train.writelines(line) if i % nfold != test_id else file_test.writelines(line)
    file_train.close()
    file_test.close()


def calculate_result(id):
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
        file.write("\n")
        file.close()
        shutil.copy("./Test_Result.txt", "./snapshot/" + mulu)
        shutil.copy("./Test_Result.txt", "./Temp_Test_Result/Test_Result_" + str(id) + ".txt")
    best_result = result[len(result) - 1]
    return best_result


# calculate the all cv means
def cal_mean(list):
    sum = 0
    for i in list:
        sum += i
    avg = sum / len(list)
    return avg


# load data
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
print("\nLoading data...")
args.cuda = (args.no_cuda) and torch.cuda.is_available()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# save file
mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.mulu = mulu
args.save_dir = os.path.join(args.save_dir, mulu)
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

if os.path.exists("./Parameters.txt"):
    os.remove("./Parameters.txt")

print("\n cpu_count \n", mu.cpu_count())
torch.set_num_threads(args.num_threads)

Temp_Test_Result = "./Temp_Test_Result"
if os.path.exists(Temp_Test_Result):
    shutil.rmtree(Temp_Test_Result)
if not os.path.isdir(Temp_Test_Result):
    os.makedirs(Temp_Test_Result)

cv_result = []

# CV loop start
for id in range(args.nfold):
    print("\nthe {} CV file".format(id))
    if args.TWO_CLASS_TASK:
        print("Executing 2 Classification Task......")
        # which data to load
        data_path = None
        if args.MR is True:
            print("loading MR data")
            data_path = args.MR_path + "/rt-polarity.all"
            cv_spilit_file(data_path, args.nfold, test_id=id)
            train_iter, dev_iter, test_iter = load_cv_data(text_field, label_field, path="./", device=args.device,
                                                           repeat=False, shuffle=args.epochs_shuffle)
        elif args.CR is True:
            print("loading CR data")
            data_path = args.CR_path + "/custrev.all"
            cv_spilit_file(data_path, args.nfold, test_id=id)
            train_iter, dev_iter, test_iter = load_cv_data(text_field, label_field, path="./", device=args.device,
                                                           repeat=False, shuffle=args.epochs_shuffle)
        elif args.Subj is True:
            print("loading Subj data")
            data_path = args.Subj_path + "/subj.all"
            cv_spilit_file(data_path, args.nfold, test_id=id)
            train_iter, dev_iter, test_iter = load_cv_data(text_field, label_field, path="./", device=args.device,
                                                           repeat=False, shuffle=args.epochs_shuffle)

        '''
        # handle external word embedding to file for convenience
        from loaddata.handle_wordEmbedding2File import WordEmbedding2File
        wordembedding = WordEmbedding2File(wordEmbedding_path="./word2vec/glove.sentiment.conj.pretrained.txt",
                                           vocab=text_field.vocab.itos, k_dim=300)
        wordembedding.handle()
        '''
    # load word2vec
    if args.word_Embedding:
        word_embedding = Word_Embedding()
        if args.embed_dim is not None:
            print("word_Embedding_Path {} ".format(args.word_Embedding_Path))
            path = args.word_Embedding_Path
        print("loading word2vec vectors...")
        word_vecs = word_embedding.load_my_vecs(path, text_field.vocab.itos, text_field.vocab.freqs, k=args.embed_dim)
        print("word2vec loaded!")
        print("num words already in word2vec: " + str(len(word_vecs)))
        print("loading unknown word2vec and convert to list...")
        print("loading unknown word by avg......")
        # word_vecs = add_unknown_words_by_uniform(word_vecs, text_field.vocab.itos, k=args.embed_dim)
        word_vecs = word_embedding.add_unknown_words_by_avg(word_vecs, text_field.vocab.itos, k=args.embed_dim)
        print("len(word_vecs) {} ".format(len(word_vecs)))
        print("unknown word2vec loaded ! and converted to list...")

    # update args and print
    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab) - 1

    # load word2vec
    if args.word_Embedding:
        args.pretrained_weight = word_vecs

    # print parameters
    print("\nParameters:")
    file = open("Parameters.txt", "a")
    file.write("\n**********   The CV = {} Parameters   **********\n".format(id))
    for attr, value in sorted(args.__dict__.items()):
        if attr.upper() != "PRETRAINED_WEIGHT" and attr.upper() != "pretrained_weight_static".upper():
            print("\t{}={}".format(attr.upper(), value))
        file.write("\t{}={}\n".format(attr.upper(), value))
    file.close()
    shutil.copy("./Parameters.txt", "./snapshot/" + mulu)
    shutil.copy("./hyperparams.py", "./snapshot/" + mulu)

    # model
    if args.CNN is True:
        print("loading CNN model.....")
        model = model_CNN.CNN_Text(args)
        # save model in this time
        shutil.copy("./models/model_CNN.py", "./snapshot/" + mulu)
    elif args.BiLSTM_1 is True:
        print("loading BiLSTM_1 model.....")
        model = model_BiLSTM_1.BiLSTM_1(args)
        # save model in this time
        shutil.copy("./models/model_BiLSTM_1.py", "./snapshot/" + mulu)
    elif args.SRU is True:
        print("loading SRU model.....")
        model = model_SRU.SRU(args)
        # save model in this time
        shutil.copy("./models/model_SRU.py", "./snapshot/" + mulu)
    elif args.BiSRU is True:
        print("loading  model.....")
        model = model_BiSRU.BiSRU(args)
        # save model in this time
        shutil.copy("./models/model_BiSRU.py", "./snapshot/" + mulu)
    elif args.SRU_Formula is True:
        print("loading SRU_Formula model.....")
        model = model_SRU_Formula.SRU_Formula(args)
        # save model in this time
        shutil.copy("./models/model_SRU_Formula.py", "./snapshot/" + mulu)
    # del args.kernel_sizes
    if args.cuda is True:
        print("using cuda......")
        model = model.cuda()
    print(model)

    if os.path.exists("./Test_Result.txt"):
        os.remove("./Test_Result.txt")

    if args.CNN is True:
        print("CNN training start......")
        model_count = train_ALL_CNN.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiLSTM_1 is True:
        print("BiLSTM_1 training start......")
        model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, args)
    elif args.SRU is True:
        print("SRU training start......")
        model_count = train_ALL_SRU.train(train_iter, dev_iter, test_iter, model, args)
    elif args.BiSRU is True:
        print("BiSRU training start......")
        model_count = train_ALL_SRU.train(train_iter, dev_iter, test_iter, model, args)
    elif args.SRU_Formula is True:
        print("SRU_Formula training start......")
        model_count = train_ALL_SRU.train(train_iter, dev_iter, test_iter, model, args)
    print("Model_count", model_count)

    # calculate the best result
    cv_result.append(calculate_result(id=id))

# calculate the result and write of all cv
print(cv_result)
cv_mean = cal_mean(cv_result)
print("The best result is {:.6f} ".format(cv_mean))
file = open("./Temp_Test_Result/Final_Result.txt", "a")
for index, value in enumerate(cv_result):
    print(index)
    stra = str(index + 1) + "   " + str(value)
    file.write(stra)
    file.write("\n")
file.write("mean_value  " + str(cv_mean))
file.close()
shutil.copytree("./Temp_Test_Result/", "./snapshot/" + mulu + "/Temp_Test_Result")


