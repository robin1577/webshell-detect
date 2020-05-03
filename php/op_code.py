#coding:utf-8
import os
import re
import subprocess
import pandas as pd
datapath="D:/webshell-detect/php/phptrain"
target_path="D:/webshell-detect/php/phptrain_opcode"
def copyfile(dir_path,target_path):
    c_dict={}
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            string=""
            cmd="php"+" -dvld.active=1 -dvld.execute=0 "+os.path.join(root,file)
            try:
                status,output=subprocess.getstatusoutput(cmd)
            except:
                status=1
                pass
            if status==0:
                string=re.findall(r'\b[A-Z_]+\b\s',output)
                string="".join(string)
                c_dict[file]=string
                del string
    dataframe=pd.DataFrame.from_dict(c_dict,orient="index")
    dataframe.to_csv("opcode.csv")
def main():
    copyfile(datapath,target_path)
main()