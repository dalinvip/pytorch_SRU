import torch
import random
torch.manual_seed(121)
random.seed(121)


# data path
Twitter_path = "./data/Twitter"
MR_path = "./data/MR"
CR_path = "./data/CR"
Subj_path = "./data/Subj"

# select which data
Twitter = True
MR = False
CR = False
Subj = False

# cv
CV = True
nfold = 10

learning_rate = 0.001
# learning_rate_decay = 0.9   # value is 1 means not change lr
learning_rate_decay = 1   # value is 1 means not change lr
epochs = 1
batch_size = 16
log_interval = 1
test_interval = 10
save_interval = 10
save_dir = "snapshot"
shuffle = True
epochs_shuffle = True
TWO_CLASS_TASK = True
dropout = 0.6
dropout_embed = 0.6
max_norm = None
clip_max_norm = 5
kernel_num = 200
kernel_sizes = "1,2,3,4"
# kernel_sizes = "5"
static = False
# model
CNN = False
LSTM = True
BiLSTM_1 = False
SRU = False
BiSRU = False
SRU_Formula = False
# select optim algorhtim to train
Adam = True
SGD = False
Adadelta = False
optim_momentum_value = 0.9
# whether to use wide convcolution True : wide  False : narrow
wide_conv = True
# min freq to include during built the vocab, default is 1
min_freq = 1
# word_Embedding
word_Embedding = False
fix_Embedding = False
embed_dim = 300
# word_Embedding_Path = "./word2vec/glove.sentiment.conj.pretrained.txt"
# word_Embedding_Path = "./word2vec/glove.840B.300d.handled.Twitter.txt"
if Twitter is True:
    word_Embedding_Path = "./word2vec/converted_word_Twitter.txt"
elif MR is True:
    word_Embedding_Path = "./word2vec/converted_word_MR.txt"
elif CR is True:
    word_Embedding_Path = "./word2vec/converted_word_CR.txt"
elif Subj is True:
    word_Embedding_Path = "./word2vec/converted_word_Subj.txt"
else:
    print("word_Embedding_Path is None")
print(word_Embedding_Path)

lstm_hidden_dim = 300
lstm_num_layers = 5
device = -1
no_cuda = False
snapshot = None
num_threads = 1
freq_1_unk = False
# whether to init w
init_weight = True
init_weight_value = 6.0
# L2 weight_decay
weight_decay = 1e-9   # default value is zero in Adam SGD
# weight_decay = 0   # default value is zero in Adam SGD
# random seed
seed_num = 233
# whether to delete the model after test acc so that to save space
rm_model = True



