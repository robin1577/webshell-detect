import os
#listdir=os.listdir("D:/webshell-detect/samples_data/WordPress")
dir_path="D:/webshell-detect/samples_data/WordPress"
extension="php"
target_path=""

def copyfile():
    for root,dirs,files in os.walk():
        for file in files:
            ext=os.path.splitext(file)[1][1:]
            if ext==extension:
                
