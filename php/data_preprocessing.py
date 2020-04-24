#coding:utf-8
import os
import subprocess
import chardet
import shutil
data=[["D:/webshell-detect/samples_data/phpT","D:/webshell-detect/php/phptrain" ,"T",".php"] ,\
    ["D:/webshell-detect/samples_data/phpF","D:/webshell-detect/php/phptrain","F",".php"]]
def copyfile(dir_path,target_path,name,extension):
    i=0
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            ext=os.path.splitext(file)[1]
            if ext==extension:
                shutil.copy(os.path.join(root,file), os.path.join(target_path,name+str(i)+extension))
                i+=1
def main():
    for sample in data:
        copyfile(sample[0],sample[1],sample[2],sample[3])
if __name__ == "__main__":
    main()