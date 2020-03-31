#codind:utf-8
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers
from keras import callbacks
import pandas as pd
path="D:\webshell-detect\php\phptrain_opcode"
max_len=200#每一个文件最大读入1000个单词
max_words=300#字典最大个数
def file_to_str(name):
    string=""
    with open(name, 'r',encoding="utf8") as f:
        line = "1"
        while line:
                try:
                    line=f.readline()
                    line=line.strip("\n")
                    line=line.strip("\r")
                    string+=line
                except:
                    line="1"
    return string
def files_to_str(path):
    t=[]
    f=[]
    tlabel=[]
    flabel=[]
    for root,dirs,files in os.walk(path):
        for name in files:
            string=file_to_str(os.path.join(root,name))
            if 'T' in name:
                t.append(string)
                tlabel.append(1)      
            elif 'F' in name:
                f.append(string)
                flabel.append(0)
    print("sum:",len(t)+len(f))
    print("True:",len(t))
    print("Talse:",len(f))
    return(t+f,tlabel+flabel)

def main():
    #生成字典
    pdata,plabels=files_to_str(path)
    tokenizer=Tokenizer(num_words=max_words,filters=""" '!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n""")#字典个数
    tokenizer.fit_on_texts(pdata)
    word_index=tokenizer.index_word
    #with open('models/tokenizer.pickle', 'wb') as f:
    #  pickle.dump(tokenizer, f)
    print(f"字典个数{len(word_index)}")
    #dataframe=pd.DataFrame.from_dict(word_index,orient="index")
    #dataframe.to_csv("test.csv")
    #文本转序列
    sequences=tokenizer.texts_to_sequences(pdata)
    
    #生成数据标签
    data=pad_sequences(sequences,maxlen=max_len,padding="post",truncating="post")
    data=data.reshape((10268,max_len,1))
    labels=np.asarray(plabels)
    labels=labels.reshape((10268,1))

    #打乱数据标签
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices]
    labels=labels[indices]
    print("data shape: ",data.shape)
    print("labes shape:",labels.shape)
    #值标准化
    #划为数据为训练,验证，测试 7:1.5:1.5
    x_train=data[:7000]
    y_train=labels[:7000]
    x_val=data[7000:8500]
    y_val=labels[7000:8500]
    x_test=data[8500:]
    y_test=labels[8500:]
    
    #构建网络
    model=models.Sequential()
    model.add(layers.Conv1D(128,6,activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(64, 6,activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(32, 6,activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    #print(model.summary())
    #编译并训练模型
    model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['acc'])
    history=model.fit(x_train,y_train,batch_size=128,epochs=200,validation_data=(x_val,y_val))
    model.save("../models/CNN_model.h5")
    #history
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(1,len(acc)+1)
    plt.plot(epochs,loss,'bo',label="Trainig loss")
    plt.plot(epochs,val_loss,'b',label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.figure()
    plt.plot(epochs,acc,'bo',label="Training acc")
    plt.plot(epochs,val_acc,'b',label="Validation acc")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.show()
    results=model.evaluate(x_test,y_test)
    print(f"损失值：{results[0]},精确度：{results[1]}")
if __name__ == "__main__":
    main()



  