import chardet
import shutil
import os
extension=".php"

data_path="D:/webshell-detect/php/phptrain"
des_path="D:/webshell-detect/php/phptrainutf8"
def change_coding(L,R):
    for root,dirs,files in os.walk(L):
        for file in files:
            ext=os.path.splitext(file)[1]
            if ext==extension:
                try:
                    coding = chardet.detect(open(os.path.join(root,file),'rb').read())['encoding']
                    with open(os.path.join(L,file), 'r', encoding=coding) as fr, open(os.path.join(R,file), 'w', encoding='utf-8') as fw:
                        content=fr.read()
                        content=str(content.encode('utf-8'),encoding='utf-8')
                        print(content,file=fw)
                except UnicodeDecodeError as e:
                    print(file+': ' + e.reason)
change_coding(data_path,des_path)