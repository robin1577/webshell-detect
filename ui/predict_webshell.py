#coding:utf-8
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers
path="D:/webshell-detect/samples_data/yii"
max_len=500#每一个文件最大读入500个单词
path='D:/webshell-detect/samples_data/yii'
def file_to_str(name):
    string=""
    with open(name, 'r',encoding="utf8") as f:
        line="1"
        while line:
            try:
                line=f.readline()
                line=line.strip("\n")
                line=line.strip("\r")
                string+=line
            except:
                line=""
    return string
def files_to_str(path,tokenizer):
    t=[]
    webshell=[]
    #读取训练好的模型
    model=models.load_model("models/embed_model.h5")
    #读取文件夹的文件
    for root,dirs,files in os.walk(path):
        for name in files:
            #将文件内容转化成字符串（也就是api序列）
            string=file_to_str(os.path.join(root,name))
            string=[string]
            #将字符串（api序列）转化成整数序列（通过字典对象）
            sequence=tokenizer.texts_to_sequences(string)
            #填充或截断整数序列为规定的值
            data=pad_sequences(sequence,maxlen=max_len,padding="post",truncating="post")
            #预测
            prediction=model.predict(data)
            #结果>0.7为恶意序列
            if float(prediction[0][0]) > 0.7:
                webshell.append(os.path.join(root,name))
            del data,prediction,sequence,string
    return webshell
def predict_file(file_path):
    with open('models/tokenizer.pickle','rb') as f:
        tokenizer=pickle.load(f) 
    model=models.load_model("models/embed_model.h5")
    string=file_to_str(file_path)
    
    pass
def predict_files(files_path):
    with open('models/tokenizer.pickle','rb') as f:
        tokenizer=pickle.load(f)  
    return(files_to_str(files_path,tokenizer))
if __name__ == "__main__":
    main()