#本脚本用于使用bert计算向量
#目前没有作用 闲置脚本


from bert_embedding import BertEmbedding
import pandas as pd
#获取数据
objMed = pd.read_csv('ct.csv')
#生成bert预训练对象
bert_embedding = BertEmbedding()
for index,rows in objMed.iterrows():
    result = bert_embedding(objMed["物品名称"])
    print(result)