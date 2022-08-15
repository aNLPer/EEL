import torch
import gensim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
# distinguish confusing charge for legal judgement prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bert(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese",
                                              num_labels=label_size,
                                              output_attentions=False,
                                              output_hidden_states=True)

        self.contraLinear = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.BatchNorm1d(4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

        self.classifyLinear = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, label_size)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
        hidden_state = torch.mean(outputs.last_hidden_state, dim=1)
        # [batch_size, hidden_size] -> [batch_size, hidden_size]
        contra_hidden = self.contraLinear(hidden_state)
        # [batch_size, hidden_size] -> [batch_size, label_size]
        classify_preds = self.classifyLinear(contra_hidden)
        # [batch_size, hidden_size]、[batch_size, label_size]
        return contra_hidden, classify_preds

class BERTLJP(nn.Module):
    def __init__(self,
                 hidden_size, # 隐藏状态size，
                 charge_label_size,
                 article_label_size,
                 penalty_label_size,
                 mode="concat"):

        super(BERTLJP, self).__init__()

        self.mode = mode
        self.charge_label_size = charge_label_size
        self.article_label_size = article_label_size
        self.penalty_label_size = penalty_label_size

        self.enc = BertModel.from_pretrained("bert-base-chinese",
                                              num_labels=charge_label_size,
                                              output_attentions=False,
                                              output_hidden_states=True)

        self.chargeAwareAtten = CharacAwareAtten(hidden_size, mode)

        self.articleAwareAtten = CharacAwareAtten(hidden_size, mode)

        self.chargeLinear = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.BatchNorm1d(4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

        self.chargePreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, charge_label_size)
        )

        self.articlePreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, article_label_size)
        )

        self.penaltyPreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, penalty_label_size)
        )


    def forward(self, input_ids, attention_mask):
        # [batch_size, seq_length, hidden_size]
        encs = self.enc(input_ids=input_ids, attention_mask=attention_mask)  # encoder 输出
        # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
        hidden_state = torch.mean(encs.last_hidden_state, dim=1)

        # [batch_size, hidden_size]
        charge_vecs = self.chargeLinear(hidden_state)                   # 获取charge_vecs
        # [batch_size, hidden_size]
        article_vecs = self.chargeAwareAtten(encs, charge_vecs)          # 获取article_vecs
        # [batch_size, hidden_size]
        penalty_vecs = self.articleAwareAtten(encs, article_vecs)        # 获取penalty_vecs

        # [batch_size, charge_label_size]
        charge_preds = self.chargePreds(charge_vecs)
        # [batch_size, article_label_size]
        article_preds = self.articlePreds(article_vecs)
        # [batch_size, penalty_label_size]
        penalty_preds = self.penaltyPreds(penalty_vecs)
        return charge_vecs, charge_preds, article_preds, penalty_preds

class GRULJP(nn.Module):
    def __init__(self,
                 charge_label_size,
                 article_label_size,
                 penalty_label_size,
                 voc_size, # 词汇表
                 pretrained_model,
                 hidden_size=128, # 隐藏状态size，
                 num_layers = 2,
                 bidirectional=True,
                 dropout=0.5,
                 mode="concat"):
        super(GRULJP, self).__init__()

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.charge_label_size = charge_label_size
        self.article_label_size = article_label_size
        self.penalty_label_size = penalty_label_size
        self.mode = mode
        self.pretrained_model = pretrained_model

        # self.em = nn.Embedding(self.voc_size, self.hidden_size, padding_idx=0)
        vectors = torch.tensor(self.pretrained_model.vectors, dtype=torch.float32).to(device)
        self.em = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.enc = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,
                          batch_first=True,
                          bidirectional=self.bidirectional)

        self.chargeAwareAtten = CharacAwareAtten(2*self.hidden_size, mode)
        self.chargeAwareAtten.to(device)
        self.articleAwareAtten = CharacAwareAtten(2*self.hidden_size, mode)
        self.articleAwareAtten.to(device)


        self.chargeLinear = nn.Sequential(
            nn.Linear(2*self.hidden_size, 4*self.hidden_size),
            nn.BatchNorm1d(4*self.hidden_size),
            nn.GELU(),
            nn.Linear(4*self.hidden_size, 2*self.hidden_size),
            # nn.Dropout(p=0.5)
        )

        self.chargePreds = nn.Sequential(
            nn.Linear(2*self.hidden_size, 4*self.hidden_size),
            nn.BatchNorm1d(4*self.hidden_size),
            nn.GELU(),
            nn.Linear(4*self.hidden_size, self.charge_label_size),
            # nn.Dropout(p=0.5)
        )

        self.articlePreds = nn.Sequential(
            nn.Linear(2*self.hidden_size, 4*self.hidden_size),
            nn.BatchNorm1d(4*self.hidden_size),
            nn.GELU(),
            nn.Linear(4*self.hidden_size, self.article_label_size),
            # nn.Dropout(p=0.5)
        )

        self.penaltyPreds = nn.Sequential(
            nn.Linear(2*self.hidden_size, 4*self.hidden_size),
            nn.BatchNorm1d(4*self.hidden_size),
            nn.GELU(),
            nn.Linear(4*self.hidden_size, self.penalty_label_size),
            # nn.Dropout(p=0.5)
        )

    def forward(self, input_ids, seq_lens):
        # [batch_size, seq_length, hidden_size]
        inputs = self.em(input_ids)

        inputs_packed = pack_padded_sequence(inputs, seq_lens, batch_first=True, enforce_sorted=False)
        # packed_output
        outputs_packed, h_n = self.enc(inputs_packed)

        # [batch_size, seq_len, 2*hidden_size]
        outputs_unpacked, unpacked_lens = pad_packed_sequence(outputs_packed, batch_first=True)
        # [batch_size, 2*hidden_size]
        outputs_sum = outputs_unpacked.sum(dim=1)
        unpacked_lens = unpacked_lens.unsqueeze(dim=1).to(device)
        outputs_mean = outputs_sum/unpacked_lens

        # [batch_size, 2*hidden_size]
        charge_vecs = self.chargeLinear(outputs_mean)
        # [batch_size, 2*hidden_size]
        article_vecs = self.chargeAwareAtten(outputs_unpacked, charge_vecs)
        # [batch_size, 2*hidden_size]
        penalty_vecs = self.articleAwareAtten(outputs_unpacked, charge_vecs)

        # [batch_size, charge_label_size]
        charge_preds = self.chargePreds(charge_vecs)
        # [batch_size, article_label_size]
        article_preds = self.articlePreds(article_vecs)
        # [batch_size, penalty_label_size]
        penalty_preds = self.penaltyPreds(penalty_vecs)

        return charge_vecs, charge_preds, article_preds, penalty_preds

class CharacAwareAtten(nn.Module):
    def __init__(self, hidden_size, mode="concat"):
        """
        :param hidden_size: 隐变量的维度
        """
        super(CharacAwareAtten, self).__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        # encoder 输出的转换矩阵
        self.transM = torch.normal(0, 1, (self.hidden_size, self.hidden_size),
                                   dtype=torch.float32,
                                   requires_grad=True).to(device)
        if mode == "sum":
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size, 2 * self.hidden_size),
                nn.BatchNorm1d(2 * self.hidden_size),
                nn.GELU(),
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.Dropout(p=0.5)
            )
        if mode == "concat":
            self.linear = nn.Sequential(
                nn.Linear(2*self.hidden_size, 4 * self.hidden_size),
                nn.BatchNorm1d(4 * self.hidden_size),
                nn.GELU(),
                nn.Linear(4 * self.hidden_size, self.hidden_size),
                nn.Dropout(p=0.5)
            )

    def forward(self, encs, vecs):
        """
        :param encs: encoder outputs --> [batch_size, seq_length, hidden_size]
        :param vecs: vec  --> [batch_size, hidden_size]
        :return:vecs for preds --> [batch_size, hidden_size]
        """
        # [batch_size, seq_length, hidden_size]
        encs = torch.matmul(encs, self.transM)

        # [batch_size, hidden_size, 1]
        vecs = torch.unsqueeze(vecs, dim=2)

        # [batch_size, seq_length, 1]
        atten_score = torch.matmul(encs, vecs)

        # [batch_size, seq_length, 1]
        atten_score = atten_score.softmax(dim=1)

        # [batch_size, hidden_size, seq_length]
        encs = encs.transpose(dim0=1, dim1=2)

        # [batch_size, hidden_size, 1]
        atten_sum = torch.matmul(encs, atten_score)

        # [batch_size, hidden_size]
        atten_sum = atten_sum.squeeze()

        if self.mode == "concat":
            # [batch_size, 2*hidden_size]
            chara_vecs = torch.concat((atten_sum, vecs.squeeze()), dim=1)
            # [batch_size, hidden_size]
            return self.linear(chara_vecs)
        if self.mode == "sum":
            # [batch_size, hidden_size]
            chara_vecs = atten_sum + vecs.squeeze()
            # [batch_size, hidden_size]
            return self.linear(chara_vecs)


    def _mask(self, tensor, tensor_length, mask_value=torch.tensor(float("-inf"))):
        """
        将padding部分的注意力分数设置为-inf
        :param tensor: [batch_size, seq_length]
        :param tensor_length: 指定了每个句子的真实长度
        :param mask_value: Default: tensor(-inf)
        :return:
        """

        pass