# coding:utf-8
import torch
import pickle
import json
import gensim
import time
import utils
import numpy as np
import configparser
import torch.nn as nn
import torch.optim as optim
from models import GRULJP, GRUBase,BERTBase
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import contras_data_loader, train_distloss_fun, penalty_constrain, ConfusionMatrix, prepare_data, data_loader, check_data, Lang, make_accu2case_dataset, load_classifiedAccus, dataset_decay
from transformers import BertModel, BertTokenizer

class gru_ljp():
    def __init__(self, device, section):
        self.device = device
        print("load model parameters...")
        self.section = section
        self.config = configparser.ConfigParser()
        self.config.read('config.cfg', encoding="utf-8")
        self.EPOCH = int(self.config.get(self.section, "EPOCH"))
        self.BATCH_SIZE = int(self.config.get(self.section, "BATCH_SIZE"))
        self.HIDDEN_SIZE = int(self.config.get(self.section, "HIDDEN_SIZE"))
        self.POSITIVE_SIZE = int(self.config.get(self.section, "POSITIVE_SIZE"))
        self.MAX_LENGTH = int(self.config.get(self.section, "MAX_LENGTH"))
        self.SIM_ACCU_NUM = int(self.config.get(self.section, "SIM_ACCU_NUM"))
        self.PENALTY_LABEL_SIZE = int(self.config.get(self.section, "PENALTY_LABEL_SIZE"))
        self.LR = float(self.config.get(self.section, "LR"))
        self.STEP = int(self.config.get(self.section, "STEP"))
        self.CHARGE_RADIUS = int(self.config.get(self.section, "CHARGE_RADIUS"))
        self.PENALTY_RADIUS = int(self.config.get(self.section, "PENALTY_RADIUS"))
        self.LAMDA = float(self.config.get(self.section, "LAMDA"))  # 刑期约束系数
        self.ALPHA = float(self.config.get(self.section, "ALPHA"))  #
        self.GRU_LAYERS = int(self.config.get(self.section, 'GRU_LAYERS'))
        self.DROPOUT_RATE = float(self.config.get(self.section, "DROPOUT_RATE"))
        self.L2 = float(self.config.get(self.section, "L2"))
        self.NUM_CYCLES = int(self.config.get(self.section, "NUM_CYCLES"))
        self.WARMUP_STEP = int(self.config.get(self.section, "WARMUP_STEP"))
        self.DATA = self.config.get(self.section, "DATA")
        self.WIKI = int(self.config.get(self.section, "WIKI"))
        if self.DATA == "SMALL":
            self.corpus_info_path = "../dataprocess/CAIL-SMALL-Lang.pkl"
            self.train_data_path = "../dataset/CAIL-SMALL/train_processed_sp.txt"
            self.valid_data_path = "../dataset/CAIL-SMALL/test_processed_sp.txt"
        if self.DATA == "LARGE":
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
        self.accu2case = make_accu2case_dataset(self.train_data_path,
                                                lang=self.lang,
                                                input_idx=0,
                                                accu_idx=1,
                                                max_length=self.MAX_LENGTH,
                                                pretrained_vec=self.pretrained_model)

        print("load accusation similarity sheet...")
        self.category2accu, self.accu2category = load_classifiedAccus(self.accu_similarity)

        print("load normal valid data...")
        self.valid_seq, self.valid_charge_labels, self.valid_article_labels, self.valid_penalty_labels = \
            prepare_data(self.valid_data_path, self.lang, input_idx=0, max_length=self.MAX_LENGTH,
                         pretrained_vec=self.pretrained_model)

        print("load normal train data...")
        self.train_seq, self.train_charge_labels, self.train_article_labels, self.train_penalty_labels = \
            prepare_data(self.train_data_path, self.lang, input_idx=0, max_length=self.MAX_LENGTH,
                         pretrained_vec=self.pretrained_model)

    def load_model(self, ablation=None):
        """
        :param ablation: 消融实验：None, lsscl, charge-aware
        :return:
        """

        print("load model...")
        self.model = GRULJP(charge_label_size=len(self.lang.index2accu),
                            article_label_size=len(self.lang.index2art),
                            penalty_label_size=self.PENALTY_LABEL_SIZE,
                            voc_size=self.lang.n_words,
                            ablation=ablation,
                            dropout=self.DROPOUT_RATE,
                            num_layers=self.GRU_LAYERS,
                            hidden_size=self.HIDDEN_SIZE,
                            pretrained_model=self.pretrained_model,
                            mode="sum")
        self.model.to(self.device)

    def train(self, mode=None):
        """
        :param mode: ["vanilla", "lsscl", "eval_batch"]
        :return:
        """
        self.load_model(ablation=mode)
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 多分类损失
        # criterion_mml = nn.MultiMarginLoss()

        # 定义优化器 AdamW由Transfomer提供,目前看来表现很好
        # optimizer = AdamW(model.parameters(), lr=LR, weight_decay=L2)
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        optimizer = optim.AdamW([{"params": self.model.em.parameters(), 'lr': 0.00001},
                                 {"params": self.model.enc.parameters(), 'weight_decay': 0.1},
                                 {"params": self.model.chargeAwareAtten.parameters()},
                                 {'params': self.model.articleAwareAtten.parameters()},
                                 {"params": self.model.chargeLinear.parameters()},
                                 {'params': self.model.chargePreds.parameters()},
                                 {'params': self.model.articlePreds.parameters()},
                                 {'params': self.model.penaltyPreds.parameters()}
                                 ], lr=self.LR, weight_decay=self.L2)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=self.WARMUP_STEP,
                                                                       num_training_steps=self.STEP,
                                                                       num_cycles=self.NUM_CYCLES)
        print("train method start...\n")

        train_loss = 0
        train_loss_records = []
        valid_loss_records = []
        valid_acc_records = {"charge": [], "article": [], "penalty": []}
        valid_mp_records = {"charge": [], "article": [], "penalty": []}
        valid_f1_records = {"charge": [], "article": [], "penalty": []}
        valid_mr_records = {"charge": [], "article": [], "penalty": []}
        for step in range(int(self.STEP/3)):
            # 随机生成一个batch
            if step % self.EPOCH == 0:
                start = time.time()
            seqs, accu_labels, article_labels, penalty_labels = contras_data_loader(accu2case=self.accu2case,
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
            article_preds_outputs = []
            penalty_preds_outputs = []
            for i in range(self.POSITIVE_SIZE):
                # [batch_size/2, hidden_size]、[batch_size/2, label_size]
                seq_lens = []
                for tensor in seqs[i]:
                    seq_lens.append(tensor.shape[0])
                padded_input_ids = pad_sequence(seqs[i], batch_first=True).to(self.device)

                charge_vecs, charge_preds, article_preds, penalty_preds = self.model(padded_input_ids, seq_lens)
                charge_vecs_outputs.append(charge_vecs)
                charge_preds_outputs.append(charge_preds)
                article_preds_outputs.append(article_preds)
                penalty_preds_outputs.append(penalty_preds)


            contra_outputs = torch.stack(charge_vecs_outputs, dim=0)  # 2 * [batch_size/posi_size, hidden_size] -> [posi_size, batch_size/posi_size, hidden_size]
            posi_pairs_dist, neg_pairs_dist = train_distloss_fun(contra_outputs, radius=self.CHARGE_RADIUS)

            # 指控分类误差
            charge_preds_outputs = torch.cat(charge_preds_outputs,dim=0)  # [posi_size, batch_size/posi_size, label_size] -> [batch_size, label_size]
            accu_labels = [torch.tensor(l) for l in accu_labels]
            accu_labels = torch.cat(accu_labels, dim=0).to(self.device)
            charge_preds_loss = self.criterion(charge_preds_outputs, accu_labels)

            # 法律条款预测误差
            article_preds_outputs = torch.cat(article_preds_outputs, dim=0)
            article_labels = [torch.tensor(l) for l in article_labels]
            article_labels = torch.cat(article_labels, dim=0).to(self.device)
            article_preds_loss = self.criterion(article_preds_outputs, article_labels)

            # 刑期预测结果约束（相似案件的刑期应该相近）
            # penalty_contrains = torch.stack(penalty_preds_outputs, dim=0).to(self.device)
            # penalty_contrains_loss = penalty_constrain(penalty_contrains, self.PENALTY_RADIUS)

            # 刑期预测误差
            penalty_preds_outputs = torch.cat(penalty_preds_outputs, dim=0)
            penalty_labels = [torch.tensor(l) for l in penalty_labels]
            penalty_labels = torch.cat(penalty_labels, dim=0).to(self.device)
            penalty_preds_loss = self.criterion(penalty_preds_outputs, penalty_labels)

            # loss = self.ALPHA * (posi_pairs_dist - neg_pairs_dist) + charge_preds_loss+article_preds_loss+ penalty_preds_loss
                   # self.LAMDA * penalty_contrains_loss
            loss = posi_pairs_dist/(posi_pairs_dist+neg_pairs_dist) + charge_preds_loss+article_preds_loss+penalty_preds_loss
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
                article_confusMat = ConfusionMatrix(len(self.lang.index2art))
                penalty_confusMat = ConfusionMatrix(self.PENALTY_LABEL_SIZE)

                # 验证模型在验证集上的表现
                self.model.eval()
                valid_loss = 0
                val_step = 0

                for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(self.valid_seq,
                                                                                                   self.valid_charge_labels,
                                                                                                   self.valid_article_labels,
                                                                                                   self.valid_penalty_labels,
                                                                                                   shuffle=False,
                                                                                                   batch_size=10 * self.BATCH_SIZE):
                    val_seq_lens = [len(s) for s in val_seq]
                    val_input_ids = [torch.tensor(s) for s in val_seq]
                    val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(self  .device)
                    with torch.no_grad():
                        val_charge_vecs, val_charge_preds, val_article_preds, val_penalty_preds = self.model(val_input_ids,
                                                                                                        val_seq_lens)

                        val_charge_preds_loss = self.criterion(val_charge_preds, torch.tensor(val_charge_label).to(self.device))
                        val_article_preds_loss = self.criterion(val_article_preds,
                                                           torch.tensor(val_article_label).to(self.device))
                        val_penalty_preds_loss = self.criterion(val_penalty_preds,
                                                           torch.tensor(val_penalty_label).to(self.device))
                        valid_loss += val_charge_preds_loss.item()
                        valid_loss += val_article_preds_loss.item()
                        valid_loss += val_penalty_preds_loss.item()

                        charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                        article_confusMat.updateMat(val_article_preds.cpu().numpy(), np.array(val_article_label))
                        penalty_confusMat.updateMat(val_penalty_preds.cpu().numpy(), np.array(val_penalty_label))

                    val_step += 1

                train_loss_records.append(train_loss / self.EPOCH)

                valid_loss_records.append(valid_loss/val_step)

                # acc
                valid_acc_records['charge'].append(charge_confusMat.get_acc())
                valid_acc_records['article'].append(article_confusMat.get_acc())
                valid_acc_records['penalty'].append(penalty_confusMat.get_acc())

                # F1
                valid_f1_records['charge'].append(charge_confusMat.getMaF())
                valid_f1_records['article'].append(article_confusMat.getMaF())
                valid_f1_records['penalty'].append(penalty_confusMat.getMaF())

                # MR
                valid_mr_records['charge'].append(charge_confusMat.getMaR())
                valid_mr_records['article'].append(article_confusMat.getMaR())
                valid_mr_records['penalty'].append(penalty_confusMat.getMaR())

                # MP
                valid_mp_records['charge'].append(charge_confusMat.getMaP())
                valid_mp_records['article'].append(article_confusMat.getMaP())
                valid_mp_records['penalty'].append(penalty_confusMat.getMaP())

                end = time.time()
                print(
                    f"Epoch: {int((step + 1) / self.EPOCH)}  Train_loss: {round(train_loss / self.EPOCH, 6)}  Valid_loss: {round(valid_loss, 6)} \n"
                    f"Charge_Acc: {round(charge_confusMat.get_acc(), 4)}  Charge_MP: {round(charge_confusMat.getMaP(), 4)}  Charge_MR: {round(charge_confusMat.getMaR(), 4)}  Charge_F1: {round(charge_confusMat.getMaF(), 4)}\n"
                    f"Article_Acc: {round(article_confusMat.get_acc(), 4)}  Article_MP: {round(article_confusMat.getMaP(), 4)}   Article_MR: {round(article_confusMat.getMaR(), 4)} Article_F1: {round(article_confusMat.getMaF(), 4)}\n"
                    f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 4)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 4)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 4)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 4)}\n"
                    f"Time: {round((end - start) / 60, 2)}min ")


                # 保存模型
                save_path = f"../dataset/checkpoints/model-at-epoch-{mode}-{self.BATCH_SIZE}-{self.SIM_ACCU_NUM}-{int((step + 1) / self.EPOCH)}.pt"
                torch.save(self.model, save_path)

                train_loss = 0

        train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
        valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
        valid_acc_records = json.dumps(valid_acc_records, ensure_ascii=False)
        valid_mp_records = json.dumps(valid_mp_records, ensure_ascii=False)
        valid_f1_records = json.dumps(valid_f1_records, ensure_ascii=False)
        valid_mr_records = json.dumps(valid_mr_records, ensure_ascii=False)
        with open(f"../training_records_{mode}_{self.BATCH_SIZE}_{self.SIM_ACCU_NUM}.txt", "w", encoding="utf-8") as f:
            f.write('train_loss_records\t' + train_loss_records + "\n")
            f.write('valid_loss_records\t' + valid_loss_records + "\n")
            f.write('valid_acc_records\t' + valid_acc_records + "\n")
            f.write('valid_mp_records\t' + valid_mp_records + "\n")
            f.write('valid_f1_records\t' + valid_f1_records + "\n")
            f.write('valid_mr_records\t' + valid_mr_records + "\n")

    def train_wo_lsscl(self, mode="lsscl"):
        """
        :param mode: ["vanilla", "lsscl", "eval_batch"]
        :return:
        """
        self.load_model(ablation=mode)
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 多分类损失
        # criterion_mml = nn.MultiMarginLoss()

        # 定义优化器 AdamW由Transfomer提供,目前看来表现很好
        # optimizer = AdamW(model.parameters(), lr=LR, weight_decay=L2)
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        optimizer = optim.AdamW([{"params": self.model.em.parameters(), 'lr': 0.00001},
                                 {"params": self.model.enc.parameters(), 'weight_decay': 0.07},
                                 {"params": self.model.chargeAwareAtten.parameters(), 'weight_decay': 0.07},
                                 {'params': self.model.articleAwareAtten.parameters(), 'weight_decay': 0.07},
                                 {"params": self.model.chargeLinear.parameters()},
                                 {'params': self.model.chargePreds.parameters()},
                                 {'params': self.model.articlePreds.parameters()},
                                 {'params': self.model.penaltyPreds.parameters()}
                                 ], lr=self.LR, weight_decay=self.L2)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=self.WARMUP_STEP,
                                                                       num_training_steps=self.STEP,
                                                                       num_cycles=self.NUM_CYCLES)
        print("train method start...\n        ")

        train_loss = 0
        train_loss_records = []
        valid_loss_records = []
        valid_acc_records = {"charge": [], "article": [], "penalty": []}
        valid_mp_records = {"charge": [], "article": [], "penalty": []}
        valid_f1_records = {"charge": [], "article": [], "penalty": []}
        valid_mr_records = {"charge": [], "article": [], "penalty": []}
        for epoch in range(60): # 60个epoch
            start = time.time()
            for seqs, accu_labels, article_labels, penalty_labels in data_loader(self.train_seq,
                                                                               self.train_charge_labels,
                                                                               self.train_article_labels,
                                                                               self.train_penalty_labels,
                                                                               shuffle=True,
                                                                               batch_size=self.BATCH_SIZE):

                # 设置模型状态
                self.model.train()

                # 优化参数的梯度置0
                optimizer.zero_grad()

                seq_lens = []
                for s in seqs:
                    seq_lens.append(len(s))
                for i in range(len(seqs)):
                    seqs[i] = torch.tensor(seqs[i])
                padded_input_ids = pad_sequence(seqs, batch_first=True).to(self.device)

                _, charge_preds, article_preds, penalty_preds = self.model(padded_input_ids, seq_lens)


                # 指控分类误差
                charge_preds_loss = self.criterion(charge_preds, torch.tensor(accu_labels).to(self.device))

                # 法律条款预测误差
                article_preds_loss = self.criterion(article_preds, torch.tensor(article_labels).to(self.device))

                # 刑期预测误差
                penalty_preds_loss = self.criterion(penalty_preds, torch.tensor(penalty_labels).to(self.device))

                loss = (charge_preds_loss + article_preds_loss+ penalty_preds_loss)/self.BATCH_SIZE

                train_loss += loss.item()

                # 反向传播计算梯度
                loss.backward()

                # 梯度裁剪防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 更新梯度
                optimizer.step()

                # 更新学习率
                scheduler.step()

            train_loss_records.append(train_loss)

            # 训练完一个EPOCH后评价模型
            # 初始化混淆矩阵
            charge_confusMat = ConfusionMatrix(len(self.lang.index2accu))
            article_confusMat = ConfusionMatrix(len(self.lang.index2art))
            penalty_confusMat = ConfusionMatrix(self.PENALTY_LABEL_SIZE)
            # 验证模型在验证集上的表现
            self.model.eval()
            valid_loss = 0

            for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(self.valid_seq,
                                                                                               self.valid_charge_labels,
                                                                                               self.valid_article_labels,
                                                                                               self.valid_penalty_labels,
                                                                                               shuffle=False,
                                                                                               batch_size=10 * self.BATCH_SIZE):
                val_seq_lens = [len(s) for s in val_seq]
                val_input_ids = [torch.tensor(s) for s in val_seq]
                val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(self.device)
                with torch.no_grad():
                    _, val_charge_preds, val_article_preds, val_penalty_preds = self.model(val_input_ids,
                                                                                                    val_seq_lens)
                    val_charge_preds_loss = self.criterion(val_charge_preds, torch.tensor(val_charge_label).to(self.device))
                    val_article_preds_loss = self.criterion(val_article_preds,
                                                       torch.tensor(val_article_label).to(self.device))
                    val_penalty_preds_loss = self.criterion(val_penalty_preds,
                                                       torch.tensor(val_penalty_label).to(self.device))

                    valid_loss += (val_charge_preds_loss.item()+val_article_preds_loss.item()+val_penalty_preds_loss.item())/10*self.BATCH_SIZE

                    charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                    article_confusMat.updateMat(val_article_preds.cpu().numpy(), np.array(val_article_label))
                    penalty_confusMat.updateMat(val_penalty_preds.cpu().numpy(), np.array(val_penalty_label))

            valid_loss_records.append(valid_loss)

            # acc
            valid_acc_records['charge'].append(charge_confusMat.get_acc())
            valid_acc_records['article'].append(article_confusMat.get_acc())
            valid_acc_records['penalty'].append(penalty_confusMat.get_acc())

            # F1
            valid_f1_records['charge'].append(charge_confusMat.getMaF())
            valid_f1_records['article'].append(article_confusMat.getMaF())
            valid_f1_records['penalty'].append(penalty_confusMat.getMaF())

            # MR
            valid_mr_records['charge'].append(charge_confusMat.getMaR())
            valid_mr_records['article'].append(article_confusMat.getMaR())
            valid_mr_records['penalty'].append(penalty_confusMat.getMaR())

            # MP
            valid_mp_records['charge'].append(charge_confusMat.getMaP())
            valid_mp_records['article'].append(article_confusMat.getMaP())
            valid_mp_records['penalty'].append(penalty_confusMat.getMaP())

            end = time.time()
            print(
                f"Epoch: {epoch}  Train_loss: {round(train_loss / self.EPOCH, 6)}  Valid_loss: {round(valid_loss, 6)} \n"
                f"Charge_Acc: {round(charge_confusMat.get_acc(), 6)}  Charge_F1: {round(charge_confusMat.getMaF(), 6)}  Charge_MR: {round(charge_confusMat.getMaR(), 6)}  Charge_MP: {round(charge_confusMat.getMaP(), 6)}\n"
                f"Article_Acc: {round(article_confusMat.get_acc(), 6)}  Article_F1: {round(article_confusMat.getMaF(), 6)}  Article_MR: {round(article_confusMat.getMaR(), 6)}  Article_MP: {round(article_confusMat.getMaP(), 6)}\n"
                f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 6)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 6)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 6)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 6)}\n"
                f"Time: {round((end - start) / 60, 2)}min ")

            # 保存模型
            save_path = f"../dataset/checkpoints/model-at-epoch-{mode}-{self.BATCH_SIZE}-{self.SIM_ACCU_NUM}.pt"
            torch.save(self.model, save_path)

            train_loss = 0

        train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
        valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
        valid_acc_records = json.dumps(valid_acc_records, ensure_ascii=False)
        valid_mp_records = json.dumps(valid_mp_records, ensure_ascii=False)
        valid_f1_records = json.dumps(valid_f1_records, ensure_ascii=False)
        valid_mr_records = json.dumps(valid_mr_records, ensure_ascii=False)
        with open(f"../training_records_{mode}_{self.BATCH_SIZE}_{self.SIM_ACCU_NUM}.txt", "w", encoding="utf-8") as f:
            f.write('train_loss_records\t' + train_loss_records + "\n")
            f.write('valid_loss_records\t' + valid_loss_records + "\n")
            f.write('valid_acc_records\t' + valid_acc_records + "\n")
            f.write('valid_mp_records\t' + valid_mp_records + "\n")
            f.write('valid_f1_records\t' + valid_f1_records + "\n")
            f.write('valid_mr_records\t' + valid_mr_records + "\n")

    def train_wo_charge_aware(self, mode="charge-aware"):
        """
        :param mode: ["vanilla", "lsscl", "eval_batch"]
        :return:
        """
        self.load_model(ablation=mode)
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 多分类损失
        # criterion_mml = nn.MultiMarginLoss()

        # 定义优化器 AdamW由Transfomer提供,目前看来表现很好
        # optimizer = AdamW(model.parameters(), lr=LR, weight_decay=L2)
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        optimizer = optim.AdamW([{"params": self.model.em.parameters(), 'lr': 0.00001},
                                 {"params": self.model.enc.parameters(), 'weight_decay': 0.1},
                                 {"params": self.model.chargeLinear.parameters()},
                                 {'params': self.model.chargePreds.parameters()},
                                 {'params': self.model.articlePreds.parameters()},
                                 {'params': self.model.penaltyPreds.parameters()}
                                 ], lr=self.LR, weight_decay=self.L2)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=self.WARMUP_STEP,
                                                                       num_training_steps=self.STEP,
                                                                       num_cycles=self.NUM_CYCLES)
        print("train method start...\n")

        train_loss = 0
        train_loss_records = []
        valid_loss_records = []
        valid_acc_records = {"charge": [], "article": [], "penalty": []}
        valid_mp_records = {"charge": [], "article": [], "penalty": []}
        valid_f1_records = {"charge": [], "article": [], "penalty": []}
        valid_mr_records = {"charge": [], "article": [], "penalty": []}
        for step in range(int(self.STEP/2)):
            # 随机生成一个batch
            if step % self.EPOCH == 0:
                start = time.time()
            seqs, accu_labels, article_labels, penalty_labels = contras_data_loader(accu2case=self.accu2case,
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
            article_preds_outputs = []
            penalty_preds_outputs = []
            for i in range(self.POSITIVE_SIZE):
                # [batch_size/2, hidden_size]、[batch_size/2, label_size]
                seq_lens = []
                for tensor in seqs[i]:
                    seq_lens.append(tensor.shape[0])
                padded_input_ids = pad_sequence(seqs[i], batch_first=True).to(self.device)

                charge_vecs, charge_preds, article_preds, penalty_preds = self.model(padded_input_ids, seq_lens)
                charge_vecs_outputs.append(charge_vecs)
                charge_preds_outputs.append(charge_preds)
                article_preds_outputs.append(article_preds)
                penalty_preds_outputs.append(penalty_preds)


            contra_outputs = torch.stack(charge_vecs_outputs, dim=0)  # 2 * [batch_size/posi_size, hidden_size] -> [posi_size, batch_size/posi_size, hidden_size]
            posi_pairs_dist, neg_pairs_dist = train_distloss_fun(contra_outputs, radius=self.CHARGE_RADIUS)

            # 指控分类误差
            charge_preds_outputs = torch.cat(charge_preds_outputs,dim=0)  # [posi_size, batch_size/posi_size, label_size] -> [batch_size, label_size]
            accu_labels = [torch.tensor(l) for l in accu_labels]
            accu_labels = torch.cat(accu_labels, dim=0).to(self.device)
            charge_preds_loss = self.criterion(charge_preds_outputs, accu_labels)

            # 法律条款预测误差
            article_preds_outputs = torch.cat(article_preds_outputs, dim=0)
            article_labels = [torch.tensor(l) for l in article_labels]
            article_labels = torch.cat(article_labels, dim=0).to(self.device)
            article_preds_loss = self.criterion(article_preds_outputs, article_labels)

            # 刑期预测结果约束（相似案件的刑期应该相近）
            # penalty_contrains = torch.stack(penalty_preds_outputs, dim=0).to(self.device)
            # penalty_contrains_loss = penalty_constrain(penalty_contrains, self.PENALTY_RADIUS)

            # 刑期预测误差
            penalty_preds_outputs = torch.cat(penalty_preds_outputs, dim=0)
            penalty_labels = [torch.tensor(l) for l in penalty_labels]
            penalty_labels = torch.cat(penalty_labels, dim=0).to(self.device)
            penalty_preds_loss = self.criterion(penalty_preds_outputs, penalty_labels)

            loss = posi_pairs_dist / (posi_pairs_dist+neg_pairs_dist) + charge_preds_loss + article_preds_loss+ penalty_preds_loss
                   # self.LAMDA * penalty_contrains_loss

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
                article_confusMat = ConfusionMatrix(len(self.lang.index2art))
                penalty_confusMat = ConfusionMatrix(self.PENALTY_LABEL_SIZE)
                # 验证模型在验证集上的表现
                self.model.eval()
                valid_loss = 0
                val_step = 0

                for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(self.valid_seq,
                                                                                                   self.valid_charge_labels,
                                                                                                   self.valid_article_labels,
                                                                                                   self.valid_penalty_labels,
                                                                                                   shuffle=False,
                                                                                                   batch_size=10 * self.BATCH_SIZE):
                    val_seq_lens = [len(s) for s in val_seq]
                    val_input_ids = [torch.tensor(s) for s in val_seq]
                    val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(self.device)
                    with torch.no_grad():
                        val_charge_vecs, val_charge_preds, val_article_preds, val_penalty_preds = self.model(val_input_ids,
                                                                                                        val_seq_lens)
                        val_charge_preds_loss = self.criterion(val_charge_preds, torch.tensor(val_charge_label).to(self.device))
                        val_article_preds_loss = self.criterion(val_article_preds,
                                                           torch.tensor(val_article_label).to(self.device))
                        val_penalty_preds_loss = self.criterion(val_penalty_preds,
                                                           torch.tensor(val_penalty_label).to(self.device))
                        valid_loss += val_charge_preds_loss.item()
                        valid_loss += val_article_preds_loss.item()
                        valid_loss += val_penalty_preds_loss.item()
                        charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                        article_confusMat.updateMat(val_article_preds.cpu().numpy(), np.array(val_article_label))
                        penalty_confusMat.updateMat(val_penalty_preds.cpu().numpy(), np.array(val_penalty_label))
                    val_step += 1

                train_loss_records.append(train_loss / self.EPOCH)

                valid_loss = valid_loss / val_step
                valid_loss_records.append(valid_loss)

                # acc
                valid_acc_records['charge'].append(charge_confusMat.get_acc())
                valid_acc_records['article'].append(article_confusMat.get_acc())
                valid_acc_records['penalty'].append(penalty_confusMat.get_acc())

                # F1
                valid_f1_records['charge'].append(charge_confusMat.getMaF())
                valid_f1_records['article'].append(article_confusMat.getMaF())
                valid_f1_records['penalty'].append(penalty_confusMat.getMaF())

                # MR
                valid_mr_records['charge'].append(charge_confusMat.getMaR())
                valid_mr_records['article'].append(article_confusMat.getMaR())
                valid_mr_records['penalty'].append(penalty_confusMat.getMaR())

                # MP
                valid_mp_records['charge'].append(charge_confusMat.getMaP())
                valid_mp_records['article'].append(article_confusMat.getMaP())
                valid_mp_records['penalty'].append(penalty_confusMat.getMaP())

                end = time.time()
                print(
                    f"Epoch: {int((step + 1) / self.EPOCH)}  Train_loss: {round(train_loss / self.EPOCH, 6)}  Valid_loss: {round(valid_loss, 6)} \n"
                    f"Charge_Acc: {round(charge_confusMat.get_acc(), 6)}  Charge_F1: {round(charge_confusMat.getMaF(), 6)}  Charge_MR: {round(charge_confusMat.getMaR(), 6)}  Charge_MP: {round(charge_confusMat.getMaP(), 6)}\n"
                    f"Article_Acc: {round(article_confusMat.get_acc(), 6)}  Article_F1: {round(article_confusMat.getMaF(), 6)}  Article_MR: {round(article_confusMat.getMaR(), 6)}  Article_MP: {round(article_confusMat.getMaP(), 6)}\n"
                    f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 6)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 6)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 6)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 6)}\n"
                    f"Time: {round((end - start) / 60, 2)}min ")

                # 保存模型
                save_path = f"../dataset/checkpoints/model-at-epoch-{mode}-{self.BATCH_SIZE}-{self.SIM_ACCU_NUM}-{int((step + 1) / self.EPOCH)}.pt"
                torch.save(self.model, save_path)

                train_loss = 0

        train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
        valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
        valid_acc_records = json.dumps(valid_acc_records, ensure_ascii=False)
        valid_mp_records = json.dumps(valid_mp_records, ensure_ascii=False)
        valid_f1_records = json.dumps(valid_f1_records, ensure_ascii=False)
        valid_mr_records = json.dumps(valid_mr_records, ensure_ascii=False)
        with open(f"../training_records_{mode}_{self.BATCH_SIZE}_{self.SIM_ACCU_NUM}.txt", "w", encoding="utf-8") as f:
            f.write('train_loss_records\t' + train_loss_records + "\n")
            f.write('valid_loss_records\t' + valid_loss_records + "\n")
            f.write('valid_acc_records\t' + valid_acc_records + "\n")
            f.write('valid_mp_records\t' + valid_mp_records + "\n")
            f.write('valid_f1_records\t' + valid_f1_records + "\n")
            f.write('valid_mr_records\t' + valid_mr_records + "\n")

    def train_base(self):
        """
        :param mode: ["vanilla", "lsscl", "eval_batch"]
        :return:
        """
        self.model = GRUBase(charge_label_size=len(self.lang.index2accu),
                                article_label_size=len(self.lang.index2art),
                                penalty_label_size=self.PENALTY_LABEL_SIZE,
                                voc_size=self.lang.n_words,
                                dropout=self.DROPOUT_RATE,
                                num_layers=self.GRU_LAYERS,
                                hidden_size=self.HIDDEN_SIZE,
                                mode="sum")
        self.model.to(self.device)
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 多分类损失
        # criterion_mml = nn.MultiMarginLoss()

        # 定义优化器 AdamW由Transfomer提供,目前看来表现很好
        # optimizer = AdamW(model.parameters(), lr=LR, weight_decay=L2)
        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        optimizer = optim.AdamW([{"params": self.model.em.parameters(), 'lr': 0.00001},
                                 {"params": self.model.enc.parameters(), 'weight_decay': 0.07},
                                 {'params': self.model.chargePreds.parameters()},
                                 {'params': self.model.articlePreds.parameters()},
                                 {'params': self.model.penaltyPreds.parameters()}
                                 ], lr=self.LR, weight_decay=self.L2)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=self.WARMUP_STEP,
                                                                       num_training_steps=self.STEP,
                                                                       num_cycles=self.NUM_CYCLES)
        print("train method start...\n        ")

        train_loss = 0
        train_loss_records = []
        valid_loss_records = []
        valid_acc_records = {"charge": [], "article": [], "penalty": []}
        valid_mp_records = {"charge": [], "article": [], "penalty": []}
        valid_f1_records = {"charge": [], "article": [], "penalty": []}
        valid_mr_records = {"charge": [], "article": [], "penalty": []}
        for epoch in range(60): # 60个epoch
            start = time.time()
            for seqs, accu_labels, article_labels, penalty_labels in data_loader(self.train_seq,
                                                                               self.train_charge_labels,
                                                                               self.train_article_labels,
                                                                               self.train_penalty_labels,
                                                                               shuffle=True,
                                                                               batch_size=self.BATCH_SIZE):

                # 设置模型状态
                self.model.train()

                # 优化参数的梯度置0
                optimizer.zero_grad()

                seq_lens = []
                for s in seqs:
                    seq_lens.append(len(s))
                for i in range(len(seqs)):
                    seqs[i] = torch.tensor(seqs[i])
                padded_input_ids = pad_sequence(seqs, batch_first=True).to(self.device)

                charge_preds, article_preds, penalty_preds = self.model(padded_input_ids, seq_lens)


                # 指控分类误差
                charge_preds_loss = self.criterion(charge_preds, torch.tensor(accu_labels).to(self.device))

                # 法律条款预测误差
                article_preds_loss = self.criterion(article_preds, torch.tensor(article_labels).to(self.device))

                # 刑期预测误差
                penalty_preds_loss = self.criterion(penalty_preds, torch.tensor(penalty_labels).to(self.device))

                loss = (charge_preds_loss + article_preds_loss+ penalty_preds_loss)/self.BATCH_SIZE

                train_loss += loss.item()

                # 反向传播计算梯度
                loss.backward()

                # 梯度裁剪防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 更新梯度
                optimizer.step()

                # 更新学习率
                scheduler.step()

            train_loss_records.append(train_loss)

            # 训练完一个EPOCH后评价模型
            # 初始化混淆矩阵
            charge_confusMat = ConfusionMatrix(len(self.lang.index2accu))
            article_confusMat = ConfusionMatrix(len(self.lang.index2art))
            penalty_confusMat = ConfusionMatrix(self.PENALTY_LABEL_SIZE)
            # 验证模型在验证集上的表现
            self.model.eval()
            valid_loss = 0

            for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(self.valid_seq,
                                                                                               self.valid_charge_labels,
                                                                                               self.valid_article_labels,
                                                                                               self.valid_penalty_labels,
                                                                                               shuffle=False,
                                                                                               batch_size=10 * self.BATCH_SIZE):
                val_seq_lens = [len(s) for s in val_seq]
                val_input_ids = [torch.tensor(s) for s in val_seq]
                val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(self.device)
                with torch.no_grad():
                    val_charge_preds, val_article_preds, val_penalty_preds = self.model(val_input_ids,
                                                                                                    val_seq_lens)
                    val_charge_preds_loss = self.criterion(val_charge_preds, torch.tensor(val_charge_label).to(self.device))
                    val_article_preds_loss = self.criterion(val_article_preds,
                                                       torch.tensor(val_article_label).to(self.device))
                    val_penalty_preds_loss = self.criterion(val_penalty_preds,
                                                       torch.tensor(val_penalty_label).to(self.device))

                    valid_loss += (val_charge_preds_loss.item()+val_article_preds_loss.item()+val_penalty_preds_loss.item())/10*self.BATCH_SIZE

                    charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                    article_confusMat.updateMat(val_article_preds.cpu().numpy(), np.array(val_article_label))
                    penalty_confusMat.updateMat(val_penalty_preds.cpu().numpy(), np.array(val_penalty_label))

            valid_loss_records.append(valid_loss)

            # acc
            valid_acc_records['charge'].append(charge_confusMat.get_acc())
            valid_acc_records['article'].append(article_confusMat.get_acc())
            valid_acc_records['penalty'].append(penalty_confusMat.get_acc())

            # F1
            valid_f1_records['charge'].append(charge_confusMat.getMaF())
            valid_f1_records['article'].append(article_confusMat.getMaF())
            valid_f1_records['penalty'].append(penalty_confusMat.getMaF())

            # MR
            valid_mr_records['charge'].append(charge_confusMat.getMaR())
            valid_mr_records['article'].append(article_confusMat.getMaR())
            valid_mr_records['penalty'].append(penalty_confusMat.getMaR())

            # MP
            valid_mp_records['charge'].append(charge_confusMat.getMaP())
            valid_mp_records['article'].append(article_confusMat.getMaP())
            valid_mp_records['penalty'].append(penalty_confusMat.getMaP())

            end = time.time()
            print(
                f"Epoch: {epoch}  Train_loss: {round(train_loss / self.EPOCH, 6)}  Valid_loss: {round(valid_loss, 6)} \n"
                f"Charge_Acc: {round(charge_confusMat.get_acc(), 6)}  Charge_F1: {round(charge_confusMat.getMaF(), 6)}  Charge_MR: {round(charge_confusMat.getMaR(), 6)}  Charge_MP: {round(charge_confusMat.getMaP(), 6)}\n"
                f"Article_Acc: {round(article_confusMat.get_acc(), 6)}  Article_F1: {round(article_confusMat.getMaF(), 6)}  Article_MR: {round(article_confusMat.getMaR(), 6)}  Article_MP: {round(article_confusMat.getMaP(), 6)}\n"
                f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 6)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 6)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 6)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 6)}\n"
                f"Time: {round((end - start) / 60, 2)}min ")

            # 保存模型
            save_path = f"../dataset/checkpoints/model-base-at-epoch.pt"
            torch.save(self.model, save_path)

            train_loss = 0

        train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
        valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
        valid_acc_records = json.dumps(valid_acc_records, ensure_ascii=False)
        valid_mp_records = json.dumps(valid_mp_records, ensure_ascii=False)
        valid_f1_records = json.dumps(valid_f1_records, ensure_ascii=False)
        valid_mr_records = json.dumps(valid_mr_records, ensure_ascii=False)
        with open(f"../training_records_base.txt", "w", encoding="utf-8") as f:
            f.write('train_loss_records\t' + train_loss_records + "\n")
            f.write('valid_loss_records\t' + valid_loss_records + "\n")
            f.write('valid_acc_records\t' + valid_acc_records + "\n")
            f.write('valid_mp_records\t' + valid_mp_records + "\n")
            f.write('valid_f1_records\t' + valid_f1_records + "\n")
            f.write('valid_mr_records\t' + valid_mr_records + "\n")

    def bertbase_for_ljp(self):
        self.BATCH_SIZE = 12
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # 实例化模型
        self.model = BERTBase(len(self.lang.index2accu), len(self.lang.index2art), self.PENALTY_LABEL_SIZE)
        self.model.to(self.device)

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 定义优化器 AdamW由Transfomer提供,目前看来表现很好
        optimizer = optim.AdamW(self.model.parameters(),
                          lr=self.LR,
                          eps=1e-8)

        # 学习率优化策略
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=500,  # Default value in run_glue.py
                                                    num_training_steps=self.STEP)
        train_losses = []
        valid_losses = []
        train_loss = 0
        valid_loss = 0
        for epoch in range(30):
            start = time.time()
            self.model.train()
            train_seqs, train_charge_labels, train_article_labels, train_penalty_labels = utils.data_loader_forBert(self.train_data_path)
            for seqs, charge_labels, article_labels, penalty_labels in data_loader(train_seqs,
                                                                                   train_charge_labels,
                                                                                   train_article_labels,
                                                                                   train_penalty_labels,
                                                                                   shuffle=True,
                                                                                   batch_size=self.BATCH_SIZE):
                # batch_start = time.time()
                optimizer.zero_grad()
                charge_labels = torch.tensor([self.lang.accu2index[l] for l in charge_labels]).to(self.device)
                article_labels = torch.tensor([self.lang.art2index[l] for l in article_labels]).to(self.device)
                penalty_labels = torch.tensor(penalty_labels).to(self.device)
                seqs = tokenizer.batch_encode_plus(seqs,
                                                   add_special_tokens=False,
                                                   max_length=400,
                                                   truncation=True,
                                                   padding=True,
                                                   return_attention_mask=True,
                                                   return_tensors='pt')
                charge_preds, article_preds, penalty_preds = self.model(seqs["input_ids"].to(self.device), seqs["attention_mask"].to(self.device))
                # 计算损失
                charge_loss = criterion(charge_preds, charge_labels)
                article_loss = criterion(article_preds, article_labels)
                penalty_loss = criterion(penalty_preds, penalty_labels)
                loss = charge_loss + article_loss + penalty_loss
                train_loss+=loss.item()/self.BATCH_SIZE

                loss.backward()

                # 梯度裁剪防止梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 更新梯度
                optimizer.step()

                # 更新学习率
                scheduler.step()
                # print(f"process one batch {time.time()-batch_start}")
            train_losses.append(train_loss)
            # torch.cuda.empty_cache()

            # print("评估模型")
            # 评估模型
            charge_confusMat = ConfusionMatrix(len(self.lang.index2accu))
            article_confusMat = ConfusionMatrix(len(self.lang.index2art))
            penalty_confusMat = ConfusionMatrix(self.PENALTY_LABEL_SIZE)
            self.model.eval()
            val_seqs, val_charge_labels, val_article_labels, val_penalty_labels = utils.data_loader_forBert(self.valid_data_path)
            for seqs, charge_labels, article_labels, penalty_labels in data_loader(val_seqs,
                                                                                   val_charge_labels,
                                                                                   val_article_labels,
                                                                                   val_penalty_labels,
                                                                                   shuffle=False,
                                                                                   batch_size=self.BATCH_SIZE):

                charge_labels = torch.tensor([self.lang.accu2index[l] for l in charge_labels]).to(self.device)
                article_labels = torch.tensor([self.lang.art2index[l] for l in article_labels]).to(self.device)
                penalty_labels = torch.tensor(penalty_labels).to(self.device)
                seqs = tokenizer.batch_encode_plus(seqs,
                                                   add_special_tokens=False,
                                                   max_length=512,
                                                   truncation=True,
                                                   padding=True,
                                                   return_attention_mask=True,
                                                   return_tensors='pt')

                with torch.no_grad():
                    charge_preds, article_preds, penalty_preds = self.model(seqs["input_ids"].to(self.device), seqs["attention_mask"].to(self.device))
                    # 计算
                    charge_loss = criterion(charge_preds.cpu(), charge_labels.cpu())
                    article_loss = criterion(article_preds.cpu(), article_labels.cpu())
                    penalty_loss = criterion(penalty_preds.cpu(), penalty_labels.cpu())
                    loss = charge_loss + article_loss + penalty_loss
                    valid_loss+=loss.item()/self.BATCH_SIZE
                charge_confusMat.updateMat(charge_preds.cpu().numpy(), charge_labels.cpu().numpy())
                article_confusMat.updateMat(article_preds.cpu().numpy(), article_labels.cpu().numpy())
                penalty_confusMat.updateMat(penalty_preds.cpu().numpy(), penalty_labels.cpu().numpy())
            valid_losses.append(valid_loss)

            print(
                f"Epoch: {int(epoch + 1)}  Train_loss: {round(train_losses[-1], 4)}  Valid_loss: {round(valid_losses[-1], 4)} \n"
                f"Charge_Acc: {round(charge_confusMat.get_acc(), 4)}  Charge_MP: {round(charge_confusMat.getMaP(), 4)}  Charge_MR: {round(charge_confusMat.getMaR(), 4)}  Charge_F1: {round(charge_confusMat.getMaF(), 4)}\n"
                f"Article_Acc: {round(article_confusMat.get_acc(), 4)}  Article_MP: {round(article_confusMat.getMaP(), 4)}   Article_MR: {round(article_confusMat.getMaR(), 4)} Article_F1: {round(article_confusMat.getMaF(), 4)}\n"
                f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 4)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 4)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 4)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 4)}\n"
                f"Time: {round((time.time() - start) / 60, 2)}min")


def verify_sim_accu():
    "验证SIM_ACCU_NUM对模型的影响"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SIM_ACCU_NUM = [1]
    for i in range(len(SIM_ACCU_NUM)):
        print(f"\n-----------------------SIM_ACCU_NUM={SIM_ACCU_NUM[i]}------------------------\n")
        ljp = gru_ljp(device=device, section="multi-task")
        ljp.SIM_ACCU_NUM = SIM_ACCU_NUM[i]
        ljp.train()
        del ljp
        torch.cuda.empty_cache()

def verify_trainset_decay():
    "训练集衰减对模型"
    decay_rate = [0.5,0.4,0.3,0.2,0.1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rate in decay_rate:
        print(f"\n-----------------------{rate}------------------------\n")
        ljp = gru_ljp(device=device, section="multi-task")
        ljp.accu2case = dataset_decay(ljp.accu2case, rate)
        ljp.train()
        del ljp
        torch.cuda.empty_cache()

def verify_LSSCL():
    "验证LSSCL对模型的影响"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ljp = gru_ljp(device=device, section="multi-task")
    ljp.train_wo_lsscl()
    del ljp
    torch.cuda.empty_cache()

def verify_chargeAware():
    "验证LSSCL对模型的影响"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ljp = gru_ljp(device=device, section="multi-task")
    ljp.train_wo_charge_aware()
    del ljp
    torch.cuda.empty_cache()

def veryfy_LSSCL_BERT():
    "验证LSSCL方法微调BERT的效果"
    pass

if __name__=="__main__":
    # train sota
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plj = gru_ljp(device=device, section="bert-base")
    plj.bertbase_for_ljp()


    # "训练集衰减对模型影响"
    # verify_trainset_decay()

    # 验证SIM_ACCU_NUM对模型的影响
    # verify_sim_accu()

    # "验证LSSCL对模型的影响"
    # verify_LSSCL()

    # "验证charge_aware对模型的影响"
    # verify_chargeAware()

    # 验证charge-aware-attention对模型的影响
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # plj = gru_ljp(device=device, section="multi-task")
    # plj.train_wo_charge_aware(mode="charge-aware")
