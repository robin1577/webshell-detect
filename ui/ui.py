import tkinter
from tkinter import ttk
import tkinter.filedialog as tkFD
import os
import sys
import random
import win32api
from win32com.shell import shell,shellcon
#import predict_webshell
number = 0
def addfiles():
    global number
    #打开文件夹
    path_dir = tkFD.askdirectory()
    for root,dirs,files in os.walk(path_dir):
        for file in files:
            probability=predict(file)
            path=str(root)+'/'+str(file)
            if probability<9:
                tree.insert('',index=10000+number,text=file,values=path)
                number += 1
    tkinter.messagebox.showinfo('提示','已检测全部文件')
    #print(path_dir)

#添加文件
def addfile():
    global number
    path_file=tkFD.askopenfilename()
    filename=os.path.split(path_file)[1]
    tree.insert("",index=number,text=filename,values=path_file)
    print(path_file)
    tkinter.messagebox.showinfo('提示','已检测全部文件')

#清空显示
def clear():
    x=tree.get_children()#返回的是下标元组
    print("children",x)
    for item in x:
        tree.delete(item)
#TODO
def predict(file):
    return random.randint(1,10)
def del_selected():
    seleced=tree.selection()
    for item in seleced:
        filename=tree.item(item,"values")[0]
        print(filename)
        shell.SHFileOperation((0,shellcon.FO_DELETE,filename,None, shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,None,None))  #删除文件到回收站
        tree.delete(item)
    tkinter.messagebox.showinfo('提示','已删除选中的webshell')

def del_all():
    x=tree.get_children()#返回的是下标元组
    print("children",x)
    for item in x:
        filename=tree.item(item,"values")[0]
        shell.SHFileOperation((0,shellcon.FO_DELETE,filename,None, shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,None,None))  #删除文件到回收站
        tree.delete(item)
    tkinter.messagebox.showinfo('提示','已删除全部webshell')
#双击文件名打开记事本
def onDBClick(event):
    item= tree.index(tree.selection()[0])
    print("select:",tree.selection())
    path=tree.item(item,"values")[0]
    print("values:",path)
    win32api.ShellExecute(0, 'open', 'notepad.exe', path, '', 1)


#初始化窗口
window = tkinter.Tk()
window.attributes("-alpha", 0.96) 
#设置窗口标题
window.title("websehll检测")
#设置窗口大小
window.geometry('800x500')
window.resizable(False, False)#固定窗体

front=('宋体',20)
width=25
tkinter.Label(window, text="webshell检测 ", font=front, width=width*2, height=4)\
    .grid(row=0,column=0,columnspan=2)
tkinter.Label(window,text='webshell文件',font=front,width=width)\
    .grid(row=1,column=0)
tkinter.Button(window, text="添加文件夹", width=width,command=addfiles).grid(row=2, column=1)
tkinter.Button(window, text="添加文件", width=width,command=addfile).grid(row=3, column=1)
tkinter.Button(window, text="删除选中websehll", width=width,command=del_selected).grid(row=4, column=1)
tkinter.Button(window, text="删除全部webshell", width=width,command=del_all).grid(row=5, column=1)
#创建树状链表，显示检测出来的webshell文件
tree=ttk.Treeview(window)
tree["selectmode"] = "extended"
tree.bind("<Double-1>", onDBClick) #<Button-1>Double
tree.grid(row=2, rowspan=5,column=0)
tkinter.Button(window, text="清空", width=width,command=clear).grid(row=6, column=1)
#b.pack()
# 第6步，主窗口循环显示
window.mainloop()
