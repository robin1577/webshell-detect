#coding:utf-8
import os
import re
import subprocess
datapath="D:/webshell-detect/php/phptrain"
target_path="D:/webshell-detect/php/phptrain_opcode"
def copyfile(dir_path,target_path):
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            string=""
            cmd="php"+" -dvld.active=1 -dvld.execute=0 "+os.path.join(root,file)
            try:
                status,output=subprocess.getstatusoutput(cmd)
                if status==0:
                    string=re.findall(r'(\s[\b[A-Z_]+\b)\s',output)
                    string="".join(string)
                    with open(os.path.join(target_path,file),mode="w") as f:
                        f.write(string)
                    del string
            except:
                pass
def main():
    copyfile(datapath,target_path)
main()