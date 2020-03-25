#codind:utf-8
import os
#from keras import models
#from keras import layers
path="D:\webshell-detect\php\phptrain"
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
trains,labels=files_to_str(path)
 