#coding:ISO-8859-1
import os
import sys
import re
import shutil
import subprocess
#listdir=os.listdir("/mnt/d/webshell-detect/samples_data/WordPress")
data=[["/mnt/d/webshell-detect/samples_data/webshell","/mnt/d/webshell-detect/php/phptrain" ,"Ta",".php"] ,\
    ["/mnt/d/webshell-detect/samples_data/WordPress","/mnt/d/webshell-detect/php/phptrain","F",".php"],\
    ["/mnt/d/webshell-detect/samples_data/Webshell2","/mnt/d/webshell-detect/php/phptrain" ,"Tb",".php"],\
    ["/mnt/d/webshell-detect/samples_data/php-webshells","/mnt/d/webshell-detect/php/phptrain" ,"Tc",".php"]]
def copyfile(dir_path,target_path,name,extension):
    i=0
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            ext=os.path.splitext(file)[1]
            if ext==extension:
                string=""
                cmd="php"+" -dvld.active=1 -dvld.execute=0 "+os.path.join(root,file)
                try:
                    subprocess.getstatusoutput(cmd)
                except:
                    print(os.path.join(root,file))
                    return
                status,output=subprocess.getstatusoutput(cmd)
                if status==0:
                    string=re.findall(r'(\s[\b[A-Z_]+\b)\s',output)
                    string="".join(string)
                    with open(os.path.join(target_path,name+str(i)+extension),mode="w") as f:
                        f.write(string)
                    i+=1
                    if i>999:
                        return
                '''
def main():
    for sample in data:
        copyfile(sample[0],sample[1],sample[2],sample[3])
if __name__ == "__main__":
    main()