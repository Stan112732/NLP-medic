#此脚本用于计算TFIDF值，并提取其中高于某个阈值的词作为种类
#目前可以计算所有TFIDF值，但尚未确定具体哪些作为种类

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd

corpus = open('ct.txt', encoding='utf-8')#获取所有医疗器材
stopwords = open('stopwords.txt', encoding='utf-8')#分词
stopword_list = [word.strip('\n') for word in stopwords.readlines()]
corpus = [list(jieba.cut(c)) for c in corpus]
print(corpus)
#引用包进行TFIDF值计算
vectorizer = TfidfVectorizer(lowercase=False, tokenizer = lambda doc: doc,stop_words=stopword_list,max_features=150)#设置stopword和获取词的阈值
tdm = vectorizer.fit_transform(corpus)
names = {'name':vectorizer.get_feature_names()}
#print(tdm.toarray().sum(axis=0))
#创建DF并将物品名与其TFIDF值赋予DF
data={'word':vectorizer.get_feature_names(),
      'tfidf':tdm.toarray().sum(axis=0).tolist()}
df=pd.DataFrame(data)
df.sort_values(by="tfidf",ascending=True)
print(df)
df.to_csv('F:/STAGE/tfidf1.csv',encoding='utf_8_sig')
