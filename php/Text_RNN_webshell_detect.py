#codind:utf-8
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers,callback,Input
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
    
def TextCNN_model_1(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test):
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 =layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 =layers.MaxPooling1D(pool_size=48)(cnn1)
    cnn2 =layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 =layers.MaxPooling1D(pool_size=47)(cnn2)
    cnn3 =layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 =layers.MaxPooling1D(pool_size=46)(cnn3)
    # 合并三个模型的输出向量
    cnn =layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat =layers.Flatten()(cnn)
    drop =layers.Dropout(0.2)(flat)
    main_output =layers.Dense(3, activation='softmax')(drop)
    model =Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10)
    #y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

    #编译并训练模型
    model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['acc'])
    history=model.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_val,y_val))
    model.save("../models/embed_model.h5")
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



  