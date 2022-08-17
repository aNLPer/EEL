# coding:utf-8
import torch
import pickle
import json
import gensim
import time
import numpy as np
import configparser
import torch.nn as nn
import torch.optim as optim
from models import GRULJP, ChargePred
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import contras_data_loader, train_distloss_fun, ConfusionMatrix, prepare_valid_data, data_loader, check_data, Lang, make_accu2case_dataset, load_classifiedAccus
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

class gru_ljp():
    def __init__(self, device, section):
        self.device = device
        self.section = section
        self.load_params()
        self.load_model()

    def load_params(self, DATA="SMALL"):
        self.config = configparser.ConfigParser()
        self.config.read('config.cfg', encoding="utf-8")
        self.EPOCH = int(self.config.get(self.section, "EPOCH"))
        self.BATCH_SIZE = int(self.config.get(self.section, "BATCH_SIZE"))
        self.HIDDEN_SIZE = int(self.config.get(self.section, "HIDDEN_SIZE"))
        self.POSITIVE_SIZE = int(self.config.get(self.section, "POSITIVE_SIZE"))
        self.MAX_LENGTH = int(self.config.get(self.section, "MAX_LENGTH"))
        self.SIM_ACCU_NUM = int(self.config.get(self.section, "SIM_ACCU_NUM"))
        self.LR = float(self.config.get(self.section, "LR"))
        self.STEP = int(self.config.get(self.section, "STEP"))
        self.CHARGE_RADIUS = int(self.config.get(self.section, "CHARGE_RADIUS"))
        self.ALPHA = float(self.config.get(self.section, "ALPHA"))  #
        self.GRU_LAYERS = int(self.config.get(self.section, 'GRU_LAYERS'))
        self.DROPOUT_RATE = float(self.config.get(self.section, "DROPOUT_RATE"))
        self.L2 = float(self.config.get(self.section, "L2"))
        self.NUM_CYCLES = int(self.config.get(self.section, "NUM_CYCLES"))
        self.WARMUP_STEP = int(self.config.get(self.section, "WARMUP_STEP"))
        self.DATA = self.config.get(self.section, "DATA")
        self.WIKI = int(self.config.get(self.section, "WIKI"))
        if DATA == "SMALL":
            self.corpus_info_path = "../dataprocess/CAIL-SMALL-Lang.pkl"
            self.train_data_path = "../dataset/CAIL-SMALL/train_processed_sp.txt"
            self.valid_data_path = "../dataset/CAIL-SMALL/test_processed_sp.txt"
        if DATA == "LARGE":
            self.corpus_info_path = "../dataprocess/Lang-CAIL-LARGE-WORD.pkl"
            self.train_data_path = "../dataset/CAIL-LARGE/train_processed_sp.txt"
            self.valid_data_path = "../dataset/CAIL-LARGE/test_processed_sp.txt"

        self.accu_similarity = "../dataprocess/accusation_classified_v2_2.txt"

        if self.WIKI == 0:
            self.pretrain_lm = f'../dataset/pretrain/law_token_vec_{self.HIDDEN_SIZE}.bin'
        else:
            self.pretrain_lm = f'../dataset/pretrain/wiki_token_vec_300.bin'

        print("load corpus info...")
        with open(self.corpus_info_path, "rb") as f:
            self.lang = pickle.load(f)

        print("load pretrained word2vec...")
        self.pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrain_lm, binary=False)

        print("load dataset classified by accusation...")
        self.accu2case = make_accu2case_dataset(self.train_data_path, lang=self.lang, input_idx=0, accu_idx=1,
                                           max_length=self.MAX_LENGTH,
                                           pretrained_vec=self.pretrained_model)

        print("load accusation similarity sheet...")
        self.category2accu, self.accu2category = load_classifiedAccus(self.accu_similarity)

    def load_model(self):
        print("load model...")
        self.model = ChargePred(charge_label_size=len(self.lang.index2accu),
                                voc_size=self.lang.n_words,
                                dropout=self.DROPOUT_RATE,
                                num_layers=self.GRU_LAYERS,
                                hidden_size=self.HIDDEN_SIZE,
                                pretrained_model=self.pretrained_model)

        self.model.to(self.device)

    def train(self):
        """
        :param mode: ["vanilla", "lsscl", "eval_batch"]
        :return:
        """
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 多分类损失
        # criterion_mml = nn.MultiMarginLoss()

        # 定义优化器 AdamW由Transfomer提供,目前看来表现很好
        # optimizer = AdamW(model.parameters(), lr=LR, weight_decay=L2)
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        optimizer = optim.AdamW([{"params": self.model.em.parameters(), 'lr': 0.0001},
                                 {"params": self.model.enc.parameters(), 'weight_decay': 0.08},
                                 {"params": self.model.chargeLinear.parameters()},
                                 {'params': self.model.chargePreds.parameters()},
                                 ], lr=self.LR, weight_decay=self.L2)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=self.WARMUP_STEP,
                                                                       num_training_steps=self.STEP,
                                                                       num_cycles=self.NUM_CYCLES)
        print("train method start......\n")

        train_loss = 0
        train_loss_records = []
        valid_loss_records = []
        valid_acc_records = []
        valid_mp_records = []
        valid_f1_records = []
        valid_mr_records = []
        for step in range(self.STEP):
            # 随机生成一个batch
            if step % self.EPOCH == 0:
                start = time.time()
            seqs, accu_labels, _, _ = contras_data_loader(accu2case=self.accu2case,
                                                        batch_size=self.BATCH_SIZE,
                                                        positive_size=self.POSITIVE_SIZE,
                                                        sim_accu_num=self.SIM_ACCU_NUM,
                                                        category2accu=self.category2accu,
                                                        accu2category=self.accu2category)
            # 设置模型状态
            self.model.train()

            # 优化参数的梯度置0
            optimizer.zero_grad()

            # 计算模型的输出
            charge_vecs_outputs = []
            charge_preds_outputs = []
            for i in range(self.POSITIVE_SIZE):
                # [batch_size/2, hidden_size]、[batch_size/2, label_size]
                seq_lens = []
                for tensor in seqs[i]:
                    seq_lens.append(tensor.shape[0])
                padded_input_ids = pad_sequence(seqs[i], batch_first=True).to(device)

                charge_vecs, charge_preds = self.model(padded_input_ids, seq_lens)
                charge_vecs_outputs.append(charge_vecs)
                charge_preds_outputs.append(charge_preds)

            # charge_vecs的对比误差
            contra_outputs = torch.stack(charge_vecs_outputs, dim=0)  # 2 * [batch_size/posi_size, hidden_size] -> [posi_size, batch_size/posi_size, hidden_size]
            posi_pairs_dist, neg_pairs_dist = train_distloss_fun(contra_outputs, radius=self.CHARGE_RADIUS)

            # 指控分类误差
            charge_preds_outputs = torch.cat(charge_preds_outputs,dim=0)  # [posi_size, batch_size/posi_size, label_size] -> [batch_size, label_size]
            accu_labels = [torch.tensor(l) for l in accu_labels]
            accu_labels = torch.cat(accu_labels, dim=0).to(device)
            charge_preds_loss = self.criterion(charge_preds_outputs, accu_labels)


            loss = self.ALPHA * (posi_pairs_dist - neg_pairs_dist) + charge_preds_loss

            train_loss += loss.item()

            # 反向传播计算梯度
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 更新梯度
            optimizer.step()

            # 更新学习率
            scheduler.step()
            # 训练完一个EPOCH后评价模型
            if (step + 1) % self.EPOCH == 0:
                # 初始化混淆矩阵
                charge_confusMat = ConfusionMatrix(len(self.lang.index2accu))
                # 验证模型在验证集上的表现
                self.model.eval()
                valid_loss = 0
                val_step = 0
                valid_seq, valid_charge_labels, valid_article_labels, valid_penalty_labels = \
                    prepare_valid_data(self.valid_data_path, self.lang, input_idx=0, max_length=self.MAX_LENGTH,
                                       pretrained_vec=self.pretrained_model)

                for val_seq, val_charge_label, _, _ in data_loader(valid_seq,
                                                                   valid_charge_labels,
                                                                   valid_article_labels,
                                                                   valid_penalty_labels,
                                                                   shuffle=False,
                                                                   batch_size=10 * self.BATCH_SIZE):

                    val_seq_lens = [len(s) for s in val_seq]
                    val_input_ids = [torch.tensor(s) for s in val_seq]
                    val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(device)
                    with torch.no_grad():
                        val_charge_vecs, val_charge_preds = self.model(val_input_ids, val_seq_lens)
                        val_charge_preds_loss = self.criterion(val_charge_preds, torch.tensor(val_charge_label).to(device))

                        valid_loss += val_charge_preds_loss.item()

                        charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                    val_step += 1

                train_loss_records.append(train_loss / self.EPOCH)

                valid_loss = valid_loss / val_step * self.BATCH_SIZE
                valid_loss_records.append(valid_loss)

                # acc
                valid_acc_records.append(charge_confusMat.get_acc())
                # F1
                valid_f1_records.append(charge_confusMat.getMaF())
                # MR
                valid_mr_records.append(charge_confusMat.getMaR())
                # MP
                valid_mp_records.append(charge_confusMat.getMaP())

                end = time.time()
                print(
                    f"Epoch: {int((step + 1) / self.EPOCH)}  Train_loss: {round(train_loss / self.EPOCH, 6)}  Valid_loss: {round(valid_loss, 6)} \n"
                    f"Charge_Acc: {round(charge_confusMat.get_acc(), 6)}  Charge_MP: {round(charge_confusMat.getMaP(), 6)}  Charge_MR: {round(charge_confusMat.getMaR(), 6)}  Charge_F1: {round(charge_confusMat.getMaF(), 6)}\n"
                    f"Time: {round((end - start) / 60, 2)}min ")

                # 保存模型
                save_path = f"../dataset/charge_pred_checkpoints/model-at-epoch-{self.BATCH_SIZE}-{self.SIM_ACCU_NUM}-{int((step + 1) / self.EPOCH)}.pt"
                torch.save(self.model, save_path)

                train_loss = 0

        train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
        valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
        valid_acc_records = json.dumps(valid_acc_records, ensure_ascii=False)
        valid_mp_records = json.dumps(valid_mp_records, ensure_ascii=False)
        valid_mr_records = json.dumps(valid_mr_records, ensure_ascii=False)
        valid_f1_records = json.dumps(valid_f1_records, ensure_ascii=False)
        with open(f"../charge_pred_records_{self.BATCH_SIZE}_{self.SIM_ACCU_NUM}.txt", "w", encoding="utf-8") as f:
            f.write('train_loss_records\t' + train_loss_records + "\n")
            f.write('valid_loss_records\t' + valid_loss_records + "\n")
            f.write('valid_acc_records\t' + valid_acc_records + "\n")
            f.write('valid_mp_records\t' + valid_mp_records + "\n")
            f.write('valid_mr_records\t' + valid_mr_records + "\n")
            f.write('valid_f1_records\t' + valid_f1_records + "\n")

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ljp = gru_ljp(device=device, section="charge-prediction")
    ljp.train()