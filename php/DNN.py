#codind:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers,callbacks
from keras import regularizers
from keras.utils  import plot_model
import pandas as pd
path="D:\webshell-detect\php\phptrain_opcode"
max_len=100#每一个文件最大读入1000个单词
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
    '''
    #生成字典
    pdata,plabels=files_to_str(path)
    tokenizer=Tokenizer(num_words=max_words,filters=""" '!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n""")#字典个数
    tokenizer.fit_on_texts(pdata)
    word_index=tokenizer.index_word
    #with open('models/tokenizer.pickle', 'wb') as f:
    #  pickle.dump(tokenizer, f)
    print(f"字典个数{len(word_index)}")
    #dataframe=pd.DataFrame.from_dict(word_index,orient="index")
    # dataframe.to_csv("../models/test.csv")
    #文本转序列
    sequences=tokenizer.texts_to_sequences(pdata)
    #生成数据标签
    data=pad_sequences(sequences,maxlen=max_len)
    labels=np.asarray(plabels)
    #打乱数据标签
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices]
    labels=labels[indices]
    print("data shape: ",data.shape)
    print("labes shape:",labels.shape)
    #值标准化
    #划为数据为训练,验证，测试 6:2:2
    x_train=data[:6000]
    y_train=labels[:6000]
    x_val=data[6000:7500]
    y_val=labels[6000:7500]
    x_test=data[7500:]
    y_test=labels[7500:]
    '''
    #构建网络
    model=models.Sequential()
    model.add(layers.Dense(64,activation="relu",input_shape=(max_len)))
    model.add(layers.Dropout(0.4))
    #model.add(layers.Dense(32,activation="relu"))
    model.add(layers.Dense(8,activation="relu"))
    #model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1,activation="sigmoid"))
    plot_model(model,show_shapes=True,to_file='../logs/DNN_model.png')
    #可视化模块
    callback=[
        callbacks.TensorBoard(
            log_dir=r"D:\webshell-detect\logs\phplog",
            histogram_freq=1,
            write_images=1,
        )
    ]
    #显示模型层

    '''
    #编译并训练模型
    model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['acc'])
    history=model.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_val,y_val),callbacks=callback)
    model.save("../models/DNN_embed_model.h5")
    #history
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label="Training acc")
    plt.plot(epochs,val_acc,'b',label="Validation acc")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs,loss,'bo',label="Trainig loss")
    plt.plot(epochs,val_loss,'b',label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()
    results=model.evaluate(x_test,y_test)
    print(f"损失值：{results[0]},精确度：{results[1]}")
    '''
if __name__ == "__main__":
    main()



  