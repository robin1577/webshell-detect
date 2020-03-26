#coding:utf-8
import sys
from PyQt5.QtWidgets import QApplication,QLabel,QWidget
from PyQt5.QtGui import QIcon
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI() #界面绘制交给InitUi方法
    def initUI(self):
        #设置窗口的位置和大小
        self.setGeometry(300, 300, 300, 220)  
        #设置窗口的标题
        self.setWindowTitle('Icon')
        #设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon('ui/icon.jpg'))        
        #显示窗口
        self.show()
def main():
    #界面实例
    app=QApplication([])
    #QWidget部件是pyqt5所有用户界面对象的基类。他为QWidget提供默认构造函数。默认构造函数没有父类。
    w=Window()
    #点击关闭窗口才关闭窗口
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()