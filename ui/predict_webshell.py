#coding:utf-8
import os
import numpy as np
import re
import subprocess
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers
path="D:/webshell-detect/samples_data/yii"
max_len=1000#每一个序列最大读入1000个操作码
max_words=300

def opcode(path):
    string=""
    cmd="php"+" -dvld.active=1 -dvld.execute=0 "+path
    try:
        status,output=subprocess.getstatusoutput(cmd)
    except:
        status=1
        pass
    if status==0:
        string=re.findall(r'\b[A-Z_]+\b\s',output)
        string="".join(string)
    return string
def wbshell_pre(string,tokenizer,model):
    data=[string]
    sequences=tokenizer.texts_to_sequences(data)
    print(sequences)
    print("*************************")
    data=pad_sequences(sequences,maxlen=max_len)
    print(data.shape)
    print("*******************")
    p=model.predict(data)
    return p
def predict_file(file_path):
    string=opcode(file_path)
    with open('../models/tokenizer.pickle','rb') as f:
        tokenizer=pickle.load(f) 
    model=models.load_model("../models/TextRNN_model.h5")
    probability=wbshell_pre(string,tokenizer,model)
    print(probability)
    return probability
if __name__ == "__main__":
    pass