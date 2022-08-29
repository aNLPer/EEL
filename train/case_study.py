import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import gensim
import numpy as np
from utils import make_accu2case_dataset
from torch.nn.utils.rnn import pad_sequence
from utils import prepare_data, data_loader, check_data, Lang

HIDDEN_SIZE = 300
train_data_path = "../dataset/CAIL-SMALL/train_processed_sp.txt"
corpus_info_path = "../dataprocess/CAIL-SMALL-Lang.pkl"
CPATH = "../dataset/case_study.txt"
MPATH = f"../dataset/checkpoints_sota/model-at-epoch-None-64-8-37-s.pt"
pretrain_lm = f'../dataset/pretrain/law_token_vec_{HIDDEN_SIZE}.bin'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("load corpus info")
f = open(corpus_info_path, "rb")
lang = pickle.load(f)
f.close()

print("load pretrained word2vec")
pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_lm, binary=False)


# print("load dataset classified by accusation")
# accu2case = make_accu2case_dataset(train_data_path, lang=lang, input_idx=0, accu_idx=1, max_length=400, pretrained_vec=pretrained_model)


arr = torch.zeros(size=(4,4))
model = torch.load(MPATH)
model.to(device)
model.eval()
valid_seq, valid_charge_labels, valid_article_labels, valid_penalty_labels = \
            prepare_data(CPATH, lang,input_idx=0, max_length=400, pretrained_vec=pretrained_model)

for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(valid_seq,
                                                                                   valid_charge_labels,
                                                                                   valid_article_labels,
                                                                                   valid_penalty_labels,
                                                                                   shuffle = False,
                                                                                   batch_size=4):
    val_seq_lens = [len(s) for s in val_seq]
    val_input_ids = [torch.tensor(s) for s in val_seq]
    val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(device)
    with torch.no_grad():
        dist = nn.PairwiseDistance(p=2)
        val_charge_vecs, val_charge_preds, val_article_preds, val_penalty_preds = model(val_input_ids, val_seq_lens)
        charge_vecs = val_charge_vecs.tolist()
        for i in range(4):
            for j in range(4):
                arr[i][j] = round(dist(val_charge_vecs[i], val_charge_vecs[j]).item(), 2)
print(arr.numpy())

v_image = plt.imshow(arr.numpy(), cmap='viridis')
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
plt.colorbar(v_image, fraction=0.046, pad=0.04)
plt.xticks(np.arange(0, 4, 1))#不显示坐标刻度
plt.yticks(np.arange(0, 4, 1))
plt.savefig("case_study---.png",bbox_inches='tight')
plt.show()