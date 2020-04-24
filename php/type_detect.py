import os
import chardet
extension=".php"
dir_path_1=r"D:/webshell-detect\samples_data\phpT"
dir_path_2=r"D:/webshell-detect\samples_data\phpF"
Encoding=[]
for root,dirs,files in os.walk(dir_path_1):
    for file in files:
        ext=os.path.splitext(file)[1]
        if ext==extension:
            encoding = chardet.detect(open(os.path.join(root,file),'rb').read())['encoding']
            if encoding not in Encoding:
                Encoding.append(encoding)
for root,dirs,files in os.walk(dir_path_2):
    for file in files:
        ext=os.path.splitext(file)[1]
        if ext==extension:
            encoding = chardet.detect(open(os.path.join(root,file),'rb').read())['encoding']
            if encoding not in Encoding:
                Encoding.append(encoding)
print(Encoding)
