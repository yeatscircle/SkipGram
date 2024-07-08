import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np
import random
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度函数
from time import time
from tqdm import tqdm


# set random seed to ensure result is reproducible
def set_random():
    import random
    np.random.seed(2004)
    torch.manual_seed(2004)
    random.seed(2004)


# config
MAX_VOCAB = 10000
window = 3
negative_sample = 5
hidden = 128
batch_size = 256
epochs = 2
lr = 1e-3
dtype = torch.FloatTensor
model_dict = 'model/embedding-128.th'
train_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device:", device)

# 全局设置
set_random()
with open('data/text8', 'r') as f:
    text = f.read()

text = text.lower().split()
# we can only use MAX_VOCAB - 1 words, we use <UNK> as a word
vocab = dict(Counter(text).most_common(MAX_VOCAB - 1))  # [name,count_num]
# the count of <UNK> is text length - other word's count
vocab['<UNK>'] = len(text) - np.sum(list(vocab.values()))

# save the mapping pair of word to index
# word: i || for i, word in enumerate(vocab.keys()) || 位置表示的分割
word2idx = {word: i for i, word in enumerate(vocab.keys())}
idx2word = {i: word for i, word in enumerate(vocab.keys())}
# 这个的维度是和count的遍历次数相关,其实就是每一个都转换为tensor然后和在一起
word_count = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_count / np.sum(word_count)

# refer to original paper
# 解释一下这个地方(subsampling): 由于在原数据中的“the”、“and”之类单词出现次数极多但是由于无词义,所以我们需要减少这部门的影响
word_freqs = word_freqs ** (3. / 4.)


class EmbeddingDataset(Dataset):
    def __init__(self, text, word2idx=word2idx, word_freqs=word_freqs):
        super(EmbeddingDataset, self).__init__()
        # 把文章所有句子尽享编码,其中找得到的直接使用word2idx中的词表,找不到的则使用<UNK>
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center = self.text_encoded[idx]
        # get words in window exception center word
        pos_idx = [i for i in range(idx - window, idx)] + [i for i in range(idx + 1, idx + window + 1)]
        pos_idx = [i % len(self.text_encoded) for i in pos_idx]
        pos_words = self.text_encoded[pos_idx]

        # get negative words
        neg_mask = torch.Tensor(self.word_freqs.clone())
        neg_mask[pos_words] = 0

        # 分别表示负采样的位置个数和是否放回采样
        neg_words = torch.multinomial(neg_mask, negative_sample * pos_words.shape[0], True)
        # check if nagetive sample failure exsists
        if len(set(pos_words.numpy().tolist()) & set(neg_words.numpy().tolist())) > 0:
            print('Need to resample')

        return center, pos_words, neg_words


# word2vec的基本讲解URL: https://blog.csdn.net/weixin_39910711/article/details/103696103
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, pos_words, neg_words):
        input_embedding = self.in_embedding(center)  # [batch_size, embedding_dim]
        # 正采样
        pos_embedding = self.out_embedding(pos_words)  # [batch_size, windows*2, embedding_dim]
        # 负采样
        neg_embedding = self.out_embedding(neg_words)  # [batch_size, windows*2*neg_num, embedding_dim]
        # unsqueeze增加维度, squeeze减少维度
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embedding_dim, 1]

        # bmm批量矩阵乘法--数学含义上的计算loss
        # 一定程度上两个点积越大说明两个的词性越大,整体在空间中的位置更接近
        pos_loss = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_szie, windows*2]
        neg_loss = torch.bmm(neg_embedding, input_embedding).squeeze(2)  # [batch_szie, windows*2*neg_num]

        # sigmoid + log
        pos_loss = F.logsigmoid(pos_loss).sum(1)
        neg_loss = F.logsigmoid(neg_loss).sum(1)
        loss = pos_loss + neg_loss
        return -loss

    def get_weight(self):
        # get weights to build an application for evaluation
        return self.in_embedding.weight.data.cpu().numpy()


def evaluate(filename, embedding_weights):
    # embedding_weights是训练之后的embedding向量
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:
        data = pd.read_csv(filename, sep="\t")

    # 建立手工计算相似度和模型计算相似度之间的集合
    human_similarity = []
    model_similarity = []

    for i in data.iloc[:, 0:2].index:
        # 地道道所有的行索引
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        # 根据行索引得到其具体的内容然后使用该内容进行得到word1和word2
        # 不在字母表即为<UNK>则跳过
        if word1 not in word2idx or word2 not in word2idx:
            continue
        else:
            word1_idx, word2_idx = word2idx[word1], word2idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            # 在分别取出这两个单词对应的embedding向量
            con_sim = sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)
            model_similarity.append(float(con_sim[0, 0]))
            # 用余弦相似度计算这两个10000维向量的相似度。这个是模型算出来的相似度
            human_similarity.append(float(data.iloc[i, 2]))
            # 这个是人类统计得到的相似度

    # 计算 human_similarity 和 model_similarity 列表之间的斯皮尔曼秩相关系数，并返回该值
    # 斯皮尔曼秩相关系数是一种衡量两个变量之间单调关系的非参数统计方法
    return scipy.stats.spearmanr(human_similarity, model_similarity)  # model_similarity


if __name__ == '__main__':
    # 模型训练
    if train_model:
        # 初始化模型与参数
        dataset = EmbeddingDataset(text)
        # 开启多线程
        dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        model = SkipGram(vocab_size=MAX_VOCAB, embedding_dim=hidden).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        start = time()
        for epoch in range(epochs):
            for step, (input_label, pos_label, neg_label) in enumerate(tqdm(dataLoader)):
                input_label = input_label.long().to(device)
                pos_label = pos_label.long().to(device)
                neg_label = neg_label.long().to(device)
                # 3 step in torch
                optim.zero_grad()
                loss = model(input_label, pos_label, neg_label).mean().to(device)
                loss.backward()
                optim.step()

                if step % 1000 == 0 and step != 0:
                    end = time()
                    print("epoch:{}, step:{}, loss:{}, in time:{:.2f}s".format(epoch, step, loss.item(), end - start))
                    start = time()

            if epoch % 2 == 0:
                embedding_weights = model.get_weight()  # 调用最终训练好的embeding词向量
                torch.save(model.state_dict(), model_dict)  # 模型保存

    else:
        # 模型评估
        model = SkipGram(MAX_VOCAB, hidden)
        model.load_state_dict(torch.load(model_dict.format(hidden)))  # 加载模型
        # 在 MEN 和 Simplex-999 数据集上做评估
        embedding_weights = model.get_weight()
        print("simlex-999", evaluate("test_data/simlex-999.txt", embedding_weights))
        print("men", evaluate("test_data/men.txt", embedding_weights))
        print("wordsim353", evaluate("test_data/wordsim353.csv", embedding_weights))
