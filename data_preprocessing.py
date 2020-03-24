import os
import shutil
#listdir=os.listdir("D:/webshell-detect/samples_data/WordPress")
data=[["D:/webshell-detect/samples_data/webshell/php/","D:/webshell-detect/php/phptrain/" ,"t",".php"] ,\
    ["D:/webshell-detect/samples_data/WordPress/","D:/webshell-detect/php/phptrain/","F",".php"]]

def copyfile(dir_path,target_path,name,extension):
    i=0
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            ext=os.path.splitext(file)[1]
            if ext==extension:
                shutil.copy(root+file,target_path+name+str(i)+extension)
                i+=1
def main():
    for sample in data:
        copyfile(sample[0],sample[1],sample[2],sample[3])
if __name__ == "__main__":
    main()