#coding:utf-8
import os
import subprocess
import re
#listdir=os.listdir("/mnt/d/webshell-detect/samples_data/WordPress")
data=[[r"D:/webshell-detect\samples_data\phpT","D:/webshell-detect/php/phptrain" ,"T",".php"] ,\
    [r"D:/webshell-detect/samples_data/phpF","D:/webshell-detect/php/phptrain","F",".php"]]
def copyfile(dir_path,target_path,name,extension):
    i=0
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            ext=os.path.splitext(file)[1]
            if ext==extension:
                with open(os.path.join(root,file),encoding="utf-8",errors="ignore") as f:
                    p=f.read()
                t=open(os.path.join(target_path,name+str(i)+extension),mode="w",encoding="utf-8",errors="ignore")
                t.write(p)
                t.close()
                del p
                i+=1
def main():
    for sample in data:
        copyfile(sample[0],sample[1],sample[2],sample[3])
if __name__ == "__main__":
    main()