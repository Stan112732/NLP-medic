#此脚本用于计算TFIDF值，并提取其中高于某个阈值的词作为种类
#目前可以计算所有TFIDF值，但尚未确定具体哪些作为种类

import jieba
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import numpy as np

corpus = open('ct.txt', encoding='utf-8')#获取所有医疗器材
stopwords = open('stopwords.txt', encoding='utf-8')#分词
stopword_list = [word.strip('\n') for word in stopwords.readlines()]
corpus = [list(jieba.cut(c)) for c in corpus]
#print(corpus)
#引用包进行TFIDF值计算
vectorizer = TfidfVectorizer(lowercase=False, tokenizer = lambda doc: doc,stop_words=stopword_list)#设置stopword和获取词的阈值
tdm = vectorizer.fit_transform(corpus)
#print(tdm.toarray().sum(axis=0))
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
features=tdm.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

km = cluster.KMeans(n_clusters=200, random_state=100,max_iter=500)
#放入上矩阵进行 kmeans 聚类
c=km.fit(features)
print(c)
t = c.labels_ # t 存储每个样本所属的簇
i = 1
while i <= len(km.labels_):
      print(i, km.labels_[i - 1])#（样本序号 簇）
      i = i + 1
print(t)#簇
print(km.inertia_)
res = pd.DataFrame(c)
res.to_csv('./res.csv')




#创建DF并将物品名与其TFIDF值赋予DF
# data={'word':vectorizer.get_feature_names(),
#       'tfidf':tdm.toarray().sum(axis=0).tolist()}
# df=pd.DataFrame(data)
# df.sort_values(by="tfidf",ascending=True)
# print(df)
# df.to_csv('F:/STAGE/objects.csv',encoding='utf_8_sig')
