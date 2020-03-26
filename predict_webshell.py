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
                line="1"
    return string
def files_to_str(path,tokenizer):
    t=[]
    model=models.load_model("models/embed_model.h5")
    for root,dirs,files in os.walk(path):
        for name in files:
            string=file_to_str(os.path.join(root,name))
            string=[string]
            sequence=tokenizer.texts_to_sequences(string)
            data=pad_sequences(sequence,maxlen=500)
            prediction=model.predict(data)
            if float(prediction[0][0]) > 0.7:
                print(os.path.join(root,name))
            del data,prediction,sequence,string
def main():
    with open('models/tokenizer.pickle','rb') as f:
        tokenizer=pickle.load(f)  
    files_to_str(path,tokenizer)
if __name__ == "__main__":
    main()