#此脚本用于计算TFIDF值，并提取其中高于某个阈值的词作为种类
#目前可以计算所有TFIDF值，但尚未确定具体哪些作为种类

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd

corpus = open('ct.txt', encoding='utf-8')#获取所有医疗器材
stopwords = open('stopwords.txt', encoding='utf-8')#分词
corpus = [list(jieba.cut(c)) for c in corpus if c not in stopwords]#清除stopwords
print(corpus)
#引用包进行TFIDF值计算
vectorizer = TfidfVectorizer(lowercase=False, tokenizer = lambda doc: doc)
tdm = vectorizer.fit_transform(corpus)
names = {'name':vectorizer.get_feature_names()}
print(tdm.toarray().sum(axis=0))
#创建DF并将物品名与其TFIDF值赋予DF
data={'word':vectorizer.get_feature_names(),
      'tfidf':tdm.toarray().sum(axis=0).tolist()}
df=pd.DataFrame(data)
df.sort_values(by="tfidf",ascending=True)#排序（无效）
print(df)
df.to_csv('F:/STAGE/tfidf.csv',encoding='utf_8_sig')
