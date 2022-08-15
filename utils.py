import re
import random
import math
import json
import numpy as np
import pickle
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    # 语料库对象
    def __init__(self, name="corpus"):
        self.name = name
        self.word2index = {"PAD":0}
        self.word2count = {}
        self.index2word = {0:"PAD"}
        # 词汇表大小
        self.n_words = 1

        self.index2accu = []
        self.accu2index = None
        self.index2art = []
        self.art2index = None

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addLabel(self, accu, article):
        if accu not in self.index2accu:
            self.index2accu.append(accu)
        if article not in self.index2art:
            self.index2art.append(article)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def update_label2index(self):
        self.accu2index = {accu: idx for idx, accu in enumerate(self.index2accu)}
        self.art2index = {art: idx for idx, art in enumerate(self.index2art)}

class ConfusionMatrix:

    def __init__(self, n_class):
        """
        混淆矩阵的每一行代表真实标签，每一列代表预测标签
        :param n_class: 类别数目
        """
        self.n_class = n_class
        self.__mat = np.zeros((self.n_class, self.n_class))
        self.n_activated_class = None
        self.class_weights = None

    def updateMat(self, preds, labels):
        """
        根据预测结果和标签更新混淆矩阵
        :param preds:
        :param labels:
        :return:
        """
        # 更新矩阵数值
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        for i in range(len(labels_flat)):
            self.__mat[labels_flat[i]][pred_flat[i]] += 1

        # 更新每个类别权重
        counts = self.__mat.sum(axis=1)
        self.class_weights = counts/np.sum(counts)

        # 计算有效类别
        self.n_activated_class = sum(counts != 0)


    def get_acc(self):
        """
        :return: 返回准确率
        """
        return self.__mat.trace()/self.__mat.sum()

    def get_recall(self, class_idx):
        """
        返回某类别的召回率
        :param class_idx:
        :return:
        """
        return self.__nomal1()[class_idx][class_idx]

    def get_precision(self, class_idx):
        """
        返回某类别的精确率
        :param class_idx:
        :return:
        """
        return self.__nomal1(dim=0)[class_idx][class_idx]

    def get_f1(self, class_idx):
        """
        返回某类别的f1值
        :param class_idx:
        :return:
        """
        recall = self.get_recall(class_idx)
        precision = self.get_precision(class_idx)
        return 2*recall*precision / (recall+precision+0.00001)

    def getMaP(self):
        """
        :return: 返回 Macro-Precision
        """
        norm_mat = self.__nomal1(dim=0)
        return norm_mat.trace()/self.n_activated_class

    def getMiP(self):
        """
        返回 Micro-Precision, 考虑到每个类别的权重适合分布不均衡数据
        :return:
        """
        pass

    def getMaR(self):
        """
        :return: 返回 Macro-Recall
        """
        norm_mat = self.__nomal1(dim=1)
        return norm_mat.trace() / self.n_activated_class

    def getMiR(self):
        """
        返回 Micro-Recall, 考虑到每个类别的权重适合分布不均衡数据
        :return:
        """
        pass

    def getMaF(self):
        """
        返回 Macro-F1
        :return:
        """
        mat_1 = self.__nomal1()
        mat_0 = self.__nomal1(dim=0)
        mat = 2*mat_0*mat_1/(mat_1+mat_0+0.00001)
        return mat.trace()/self.n_activated_class


    def getMiF(self):
        pass

    def __nomal1(self, dim=1):
        """
        归一化矩阵
        :param dim: 指定在哪个维度做归一化 default：1
        :return:
        """
        assert dim == 1 or dim == 0
        if dim == 1:
            return self.__mat / (self.__mat.sum(axis=dim)[:, np.newaxis]+0.00001)
        else:
            return self.__mat / ((self.__mat.sum(axis=dim)+0.000001))

def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

# 大写数字转阿拉伯数字
def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)

# 过滤掉值小于100的项目
def filter_dict(data_dict, bound=100):
    return {k: v for k, v in data_dict.items() if v >= bound}

# 对字典中的每个项目求和
def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

# 字典重置
def reset_dict(data_dict):
    return {k: 0 for k, v in data_dict.items()}

# 加载停用词表、特殊符号表、标点
def get_filter_symbols(filepath):
    '''
    根据mode加载标点、特殊词或者停用词
    :param mode:
    :return:list
    '''
    return list(set([line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]))

# 生成acc2desc字典
def get_acc_desc(file_path):
    acc2desc = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dict = json.loads(line)
            if dict["accusation"] not in acc2desc:
                acc2desc[dict["accusation"]] = dict["desc"]
    return acc2desc


# 获取batch
def contras_data_loader(accu2case,
                         batch_size,
                         lang,
                         positive_size=2,
                         sim_accu_num=2,
                         ablation = False,
                         category2accu=None,
                         accu2category=None):
    """
    1. 先从accu2case中抽取出batch_size/2种不同的指控
    2. 对于每个指控随机抽取positive_size个案件

    :param batch_size:  = positive_size * sim_accu_num * sample_accus
    :param positive_size: 正样本数量
    :param accu2case: 指控：案件s （字典）
    :param category2accu: 类别：[指控s]（字典）
    :param accu2category: 指控：[类别s]（字典）
    :param sim_accu_num: 一个batch中任意指控及其相似指控的和

    其中 batch_size/sim_accu_num 为整数
    :return:
    """
    accu_labels = []
    article_labels = []
    penalty_labels = []
    seq = []
    for _ in range(positive_size):
        seq.append([])
        accu_labels.append([])
        article_labels.append([])
        penalty_labels.append([])
    # 获取数据集中的所有指控
    accus = np.array(list(accu2case.keys()))
    # 选取指控
    sample_accus = list(np.random.choice(accus, size=int(batch_size/(positive_size*sim_accu_num)), replace=False))
    selected_accus = sample_accus.copy()
    if ablation == False:
        count = 0
        while count<sim_accu_num-1:
            for accu in sample_accus:
                # 获取相似指控
                sim_accu_ = [category2accu[c] for c in accu2category[accu]]
                temp = []
                for l in sim_accu_:
                    # 筛选出在数据集中出现的相似指控
                    for i in l:
                        if i in lang.accu2index:
                            temp.append(i)
                    # temp.extend([i for i in l and i in accus])
                # 去除相似指控与selected_accus指控中的重复指控
                temp = set(temp)
                temp = temp.difference(set(selected_accus))
                # 添加不重复的相似指控
                sim_accu = list(temp)
                if len(sim_accu) != 0:
                    selected_accus.extend(np.random.choice(sim_accu, size=1))
            count+=1
    # 若获取的指控不足则随机挑选补充
    if len(selected_accus) < batch_size / positive_size:
        bias = int(batch_size/positive_size-len(selected_accus))
        selected_accus.extend(np.random.choice(list(set(accus).difference(set(selected_accus))), size=bias, replace=False))
    # print(len(set(selected_accus)))
    # 根据指控获取batch
    for accu in selected_accus:
        selected_cases_ids = np.random.choice(range(len(accu2case[accu])), size=positive_size, replace=False)
        selected_cases = [accu2case[accu][id] for id in selected_cases_ids]
        for i in range(positive_size):
            seq[i].append(torch.tensor(selected_cases[i][0], dtype=torch.long))
            accu_labels[i].append(selected_cases[i][1])
            article_labels[i].append(selected_cases[i][2])
            penalty_labels[i].append(selected_cases[i][3])

    return seq, accu_labels, article_labels, penalty_labels


def data_loader(seq, charge_labels, article_labels, penalty_labels,shuffle, batch_size):
    num_examples = len(seq)
    indices = list(range(num_examples))
    if shuffle:
        random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        ids = indices[i: min(i + batch_size, num_examples)]  # 最后⼀次可能不⾜⼀个batch
        yield [seq[j] for j in ids], \
              [charge_labels[j] for j in ids], \
              [article_labels[j] for j in ids], \
              [penalty_labels[j] for j in ids]

def train_distloss_fun(outputs, radius = 10):
    """
    :param outputs: [posi_size, batch_size/posi_size, hidden_dim]
    :param label_rep:
    :param label:
    :return:
    """
    posi_size = outputs.shape[0]
    batch_size = outputs.shape[1]
    # 正样本距离
    posi_pairs_dist =0
    for i in range(posi_size-1):
        for j in range(i+1, posi_size):
            posi_pairs_dist += torch.sum(F.pairwise_distance(outputs[i], outputs[j]))

    # 负样本距离
    # [posi_size, batch_size/2, hidden_dim] -> [batch_size/2, posi_size,  hidden_dim]
    outputs = torch.transpose(outputs, dim0=0, dim1=1)
    neg_pairs_dist = 0
    for i in range(int(0.5*batch_size)-1):
        for j in range(i+1, int(0.5*batch_size)):
            # outputs[i] outputs[j]
            for k in range(posi_size):
                dist = F.pairwise_distance(outputs[i][k], outputs[j])
                zero = torch.zeros_like(dist)+0.00001
                dist = dist.where(dist<radius, zero)
                neg_pairs_dist += torch.sum(dist)

    return posi_pairs_dist/batch_size, \
           neg_pairs_dist/batch_size

def penalty_constrain(outputs, radius = 10):
    """
        :param outputs: [posi_size, batch_size/posi_size, hidden_dim]
        :param radius: 大于radius的预测刑期优化
        :return:
        """
    posi_size = outputs.shape[0]
    batch_size = outputs.shape[1]
    y = torch.zeros(outputs.shape[1]).to(device)
    # 正样本距离
    penalty_constrain_loss = 0
    for i in range(posi_size - 1):
        for j in range(i + 1, posi_size):
            dist = F.pairwise_distance(outputs[i], outputs[j])
            penalty_constrain_loss += torch.sum(torch.where(dist>radius, dist, y))
    return penalty_constrain_loss/batch_size

def accumulated_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat), len(labels_flat)

def genConfusMat(confusMat, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    for i in range(len(labels_flat)):
        confusMat[labels_flat[i]][pred_flat[i]] += 1

def prepare_valid_data(resourcefile, lang, input_idx, max_length, pretrained_vec=None):
    seq = []
    charge_labels = []
    article_labels = []
    penaty_labels = []
    with open(resourcefile, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if pretrained_vec is not None:
                case = [pretrained_vec.get_index(w) if w in pretrained_vec.key_to_index.keys()
                        else pretrained_vec.get_index("") for w in item[input_idx]]
            else:
                case = [lang.word2index[w] for w in item[input_idx]]

            if len(case)<=max_length:
                case_clip = case
            else:
                case_clip = case[-max_length:]
            seq.append(case_clip)
            charge_labels.append(lang.accu2index[item[1]])
            article_labels.append(lang.art2index[item[2]])
            penaty_labels.append(item[3])
    return seq, charge_labels,  article_labels, penaty_labels

def check_data(lang, seq, c_label, a_label, p_label):
    seq_w = []
    for s in seq:
        s = s.tolist()
        seq_w.append("".join([lang.index2word[i] for i in s]))
    c_label_w = [lang.index2accu[i] for i in c_label]
    a_label_w = [lang.index2art[i] for i in a_label]
    return seq_w, c_label_w, a_label_w

# law内容过滤
def filterStr(law):
    # 删除第一个标点之前的内容
    pattern_head_content = re.compile(r".*?[，：。,:.]")
    head_content = pattern_head_content.match(law)
    if head_content is not None:
        head_content_span = head_content.span()
        law = law[head_content_span[1]:]

    # 删除“讼诉机关认为，......”
    pattern_3 = re.compile(r"[，。]公诉机关")
    content = pattern_3.search(law)
    if content is not None:
        content_span = content.span()
        law = law[:content_span[0]+1]

    # 删除"。...事实，"
    pattern_3 = re.compile(r"。.{2,8}事实，")
    content = pattern_3.search(law)
    if content is not None:
        content_span = content.span()
        law = law[:content_span[0]]

    # 删除括号及括号内的内容
    pattern_bracket = re.compile(r"[<《【\[(（〔].*?[〕）)\]】》>]")
    law = pattern_bracket.sub("", law)

    return law

def make_accu2case_dataset(filename, lang, input_idx, accu_idx, max_length, pretrained_vec=None):
    accu2case = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if pretrained_vec is not None:
                case = [pretrained_vec.get_index(w) if w in pretrained_vec.key_to_index.keys()
                        else pretrained_vec.get_index("") for w in item[input_idx]]
            else:
                case = [lang.word2index[w] for w in item[input_idx]]

            if len(case)<=max_length:
                case_clip = case
            else:
                case_clip = case[-max_length:]

            if item[accu_idx] not in accu2case:
                accu2case[item[accu_idx]] = [[case_clip,lang.accu2index[item[accu_idx]], lang.art2index[item[accu_idx+1]], item[accu_idx+2]]]
            else:
                accu2case[item[accu_idx]].append([case_clip,lang.accu2index[item[accu_idx]], lang.art2index[item[accu_idx+1]], item[accu_idx+2]])

    return accu2case

# def word2Index(file_path, lang, acc2id):
#     # 数据集
#     fi = open(file_path, "r", encoding="utf-8")
#     fo = open(os.path.join(data_base_path, "data_train_forModel.txt"), "w", encoding="utf-8")
#     count = 0
#     for line in fi:
#         count += 1
#         # 文本数据
#         item = json.loads(line)
#         # 将文本妆化成索引
#         item_num = []
#
#         fact_num = []
#         for fact in item[0:3]:
#             fact_num.append([lang.word2index[word] for word in fact])
#         item_num.append(fact_num[0])
#         item_num.append(fact_num[1])
#         item_num.append(fact_num[2])
#         item_num.append(acc2id[item[3].strip()])
#         item_num.append([lang.word2index[word] for word in item[4]])
#         # 序列化并写入
#         item_num_str = json.dumps(item_num, ensure_ascii=False)
#         fo.write(item_num_str+"\n")
#         if count%5000==0:
#             print(f"已处理{count}条数据")
#     fi.close()
#     fo.close()

# 统计文本长度
def sample_length(path):
    max_length = 0
    max_length_sample = 0
    min_length = float("inf")
    min_length_sample = 0
    f = open(path, "r", encoding="utf-8")
    # all, 0-20, 20-50, 50-100, 100-200, 200-500, 500-1000, 1000-2000, 2000-5000, 5000-
    count = {"all":0, "0-20":0, "20-50":0,"50-100":0,"100-200":0,
             "200-500":0,"500-1000":0,"1000-2000":0,"2000-5000":0,
             "5000-":0}
    for line in f:
        count["all"] += 1
        sample = json.loads(line)
        length = len(sample[0])
        if length > max_length:
            max_length = length
            max_length_sample = count["all"]
        if length < min_length:
            min_length = length
            min_length_sample = count["all"]
        # 长度范围统计
        if length>=5000:
            count["5000-"] += 1
        else:
            if length>=200: # 200-5000
                if length>=1000: # 1000-5000
                    if length>=2000:
                        count["2000-5000"]+=1
                    else:
                        count["1000-2000"]+=1
                else: # 200-1000
                    if length>=500:
                        count["500-1000"]+=1
                    else:
                        count["200-500"]+=1
            else: # 0-200
                if length>=50: # 50-200
                    if length>=100:
                        count["100-200"]+=1
                    else:
                        count["50-100"]+=1
                else:
                    if length>=20:
                        count["20-50"]+=1
                    else:
                        count["0-20"]+=1
    f.close()
    return min_length, min_length_sample, max_length, max_length_sample, count

# 按照指控类别统计案件分布
def sample_categories_dis(file_path):
    f = open(file_path, "r", encoding="utf-8")
    acc_dict = {}
    for line in f:
        sample = json.loads(line)
        sample_acc = sample[3]
        if sample_acc not in acc_dict:
            acc_dict[sample_acc] = 1
        else:
            acc_dict[sample_acc] += 1
    f.close()
    return acc_dict

# load accusation classified
def load_classifiedAccus(filename):
    category2accu = {}
    accu2category = {}
    with open(filename, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            count+=1
            line = line.strip()
            item = line.split(" ")
            if item[0] not in category2accu:
                category2accu[item[0]] = item[1:]
            for accu in item[1:]:
                if accu not in accu2category:
                    accu2category[accu] = [item[0]]
                else:
                    accu2category[accu].append(item[0])
    return category2accu, accu2category

def val_test_datafilter(resourcefile, targetflie):
    # 根据训练数据过滤val和test数据集
    lang_f = open("lang_data_train_preprocessed.pkl", "rb")
    lang = pickle.load(lang_f)
    lang_f.close()
    fw = open(targetflie, "w", encoding="utf-8")
    print("start filter data......")
    with open(resourcefile, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            if len(example["meta"]["accusation"]) != 1:
                continue
            example_accu = example["meta"]["accusation"][0]
            if example_accu not in lang.label2index:
                continue
            else:
                # example_fact = example["fact"]
                # example_accu_idx = lang.label2index[example_accu]
                # example_str = json.dumps([example_fact, example_accu_idx],ensure_ascii=False)
                fw.write(line)
    fw.close()
    print("processing end ......")