#coding=utf8
import sys, locale
s = "小甲"
print(s)
print(type(s))
print(sys.getdefaultencoding())
print(locale.getdefaultlocale())
"""
with open("utf2","w",encoding = "utf-8") as f:
    f.write(s)
with open("gbk2","w",encoding = "gbk") as f:
    f.write(s)
with open("jis2","w",encoding = "shift-jis") as f:
    f.write(s)
"""