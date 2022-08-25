import torch
import pickle
import gensim
from utils import make_accu2case_dataset, Lang,data_loader_cycle
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np
pred = np.array([1,0,0,1,0])
label = np.array([1,0,1,0,1])
acc = accuracy_score(label, pred)
recall = recall_score(label, pred, average="macro")
print(acc, recall)




# train_data_path = "./dataset/CAIL-SMALL/test_processed_sp.txt"
# corpus_info_path = "./dataprocess/CAIL-SMALL-Lang.pkl"
# with open(corpus_info_path, "rb") as f:
#     lang = pickle.load(f)
# MAX_LENGTH = 300
#
# pretrain_lm = "./dataset/pretrain/law_token_vec_300.bin"
# pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_lm, binary=False)
#
#
# accu2case = make_accu2case_dataset(train_data_path,
#                                 lang=lang,
#                                 input_idx=0,
#                                 accu_idx=1,
#                                 max_length=MAX_LENGTH,
#                                 pretrained_vec=pretrained_model)
#
# count = 0
# for seqs, accus, articles, penaltys in data_loader_cycle(accu2case):
#     count+=1
# print(count)

