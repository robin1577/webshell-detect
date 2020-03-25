import os
import shutil
#listdir=os.listdir("D:/webshell-detect/samples_data/WordPress")
data=[["D:/webshell-detect/samples_data/webshell","D:/webshell-detect/php/phptrain" ,"Ta",".php"] ,\
    ["D:/webshell-detect/samples_data/WordPress","D:/webshell-detect/php/phptrain","F",".php"],\
    ["D:/webshell-detect/samples_data/Webshell2","D:/webshell-detect/php/phptrain" ,"Tb",".php"],\
    ["D:/webshell-detect/samples_data/php-webshells","D:/webshell-detect/php/phptrain" ,"Tc",".php"]]
def copyfile(dir_path,target_path,name,extension):
    i=0
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            ext=os.path.splitext(file)[1]
            if ext==extension:
                shutil.copy(os.path.join(root,file),os.path.join(target_path,name+str(i)+extension))
                i+=1
                if i>999:
                    break
def main():
    for sample in data:
        copyfile(sample[0],sample[1],sample[2],sample[3])
if __name__ == "__main__":
    main()