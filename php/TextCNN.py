#codind:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers,callbacks,Input,Model
import pandas as pd
from keras.utils  import plot_model
path="D:\webshell-detect\php\phptrain_opcode"
max_len=100#每一个文件最大读入100个单词
max_words=300#字典最大个数
epoch=50
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
def TextCNN_model(x_train,y_train,x_val,y_val,x_test,y_test):
    text_input = Input(shape=(max_len,), dtype='float64')
    # 词嵌入训练
    embedding =layers.Embedding(max_words,50, input_length=max_len)(text_input)
    # 词窗大小分别为3,4,5
    cnn1 =layers.Conv1D(128, 3,padding='same', activation='relu')(embedding)
    cnn1 =layers.MaxPooling1D(3)(cnn1)
    cnn2 =layers.Conv1D(128, 4, padding='same', strides=1, activation='relu')(embedding)
    cnn2 =layers.MaxPooling1D(3)(cnn2)
    cnn3 =layers.Conv1D(128, 5, padding='same', strides=1, activation='relu')(embedding)
    cnn3 =layers.MaxPooling1D(3)(cnn3)
    # 合并三个模型的输出向量
    cnn =layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat =layers.Flatten()(cnn)
    drop =layers.Dropout(0.2)(flat)
    text_putput =layers.Dense(1, activation='sigmoid')(drop)
    model =Model(inputs=text_input, outputs=text_putput)
    plot_model(model,to_file='../logs/TextCNN_model.png')
    #print(model.summary())
    #可视化模块
    callback=[
        callbacks.TensorBoard(
            log_dir=r"D:\webshell-detect\logs\TextCNNlog",
            histogram_freq=1,
            write_images=1,
        )
    ]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history=model.fit(x_train,y_train,batch_size=128,epochs=epoch,validation_data=(x_val,y_val),callbacks=callback)
    model.save("../models/TestRNN_model.h5")
    results=model.evaluate(x_test,y_test)
    print(f"损失值：{results[0]},精确度：{results[1]}")
    return history
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
    #运行模型
    history=TextCNN_model(x_train,y_train,x_val,y_val,x_test,y_test)
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
if __name__ == "__main__":
    main()



  