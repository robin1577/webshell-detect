#codind:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers
path="D:\webshell-detect\php\phptrain"
max_len=500#每一个文件最大读入500个单词
max_words=10000#字典最大个数
def file_to_str(name):
    string=""
    with open(name,'r',encoding='ISO-8859-1') as f:
        for line in f:
            line=line.strip("\n")
            line=line.strip("\r")
            string+=line
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
    tokenizer=Tokenizer(max_words)#字典个数
    tokenizer.fit_on_texts(pdata)
    word_index=tokenizer.index_word
    print(f"字典个数{len(word_index)}")
    #文本转序列
    sequences=tokenizer.texts_to_sequences(pdata)
    #生成数据标签
    data=pad_sequences(sequences,maxlen=max_len)
    labels=np.asarray(layers)
    #打乱数据标签
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices:]
    labels=labels[indices]
    print("data shape: ",data.shape)
    print("labes shape:",labels.shape)
    #划为数据为训练,验证，测试 6:2:2
    x_train=data[:1020]
    y_train=labels[:1020]
    x_val=data[1020:1360]
    y_val=labels[1020:1360]
    x_test=data[1360:]
    y_test=labels[1360:]
    #构建网络
    model=models.Sequential()
    model.add(layers.Embedding(max_words,32,input_length=max_len))
    model.add(layers.Flatten())
    model.add(layers.Dense(32,activation=""))
    model.add(layers.Dense(1,activation="sigmoid"))
    #编译并训练模型
    model.compile(optimizer="rmsprop",loss='binary_crossentropy',metrics=['acc'])
    history=model.fit(x_train,y_train,batch_size=128,epochs=5,validation_data=(x_val,y_val))
    model.save_weights("embed_model.h5")
    #history
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label="Training acc")
    plt.plot(epochs,val_acc,'bo',label="Validation acc")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs,loss,'bo',label="Trainig loss")
    plt.plot(epochs,val_loss,'bo',label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()
    #绘制结果
    import matplotlib.pyplot as plt
if __name__ == "__main__":
    main()



 