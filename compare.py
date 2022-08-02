#此脚本用于比较其他所有物品与TFIDF.py中提取的种类的相似度
#尚未完善
import pandas as pd

data = pd.read_csv('F:/STAGE/tfidf.csv')

name = data['word']
tfidf = data['tfidf']
df = pd.DataFrame({'name':name,'tfidf':tfidf})
print(df)