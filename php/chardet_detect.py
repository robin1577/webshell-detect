import os
import chardet
extension=".php"
dir_path_1=r"D:\webshell-detect\php\phptrain"
#dir_path_2=r"D:/webshell-detect\samples_data\phpF"
Encoding=[]
i=0
for root,dirs,files in os.walk(dir_path_1):
    for file in files:
        ext=os.path.splitext(file)[1]
        i+=1
        if i>3000:break
        if ext==extension:
            encoding = chardet.detect(open(os.path.join(root,file),'rb').read())['encoding']
            if encoding not in Encoding:
                Encoding.append(encoding)
print(Encoding)
