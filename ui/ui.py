#coding:utf-8
import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QMessageBox,
    QPushButton, QApplication,QTextEdit,QHBoxLayout, QVBoxLayout,QDesktopWidget)
from PyQt5.QtGui import QIcon
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI() #界面绘制交给InitUi方法
    def initUI(self):
        #设置窗口的位置和大小
        self.resize(800,450)
        self.center() 
        #设置窗口的标题
        self.setWindowTitle('Icon')
        #设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon("icon.jpg"))        
        #开始布局：
        add_files_button=QPushButton("添加文件夹")
        add_files_button.resize(150,50)
        add_file_button=QPushButton("添加文件")
        add_file_button.resize(150,50)
        del_files_button=QPushButton("删除全部")
        del_files_button.resize(150,50)
        del_file_button=QPushButton("删除")
        del_file_button.resize(150,50)
        qh=QHBoxLayout()
        qh.addWidget(QTextEdit())
        qv=QVBoxLayout()
        qv.addWidget(add_files_button)
        qv.addWidget(add_file_button)
        qv.addWidget(del_files_button)
        qv.addWidget(del_file_button)
        qh.addLayout(qv)
        self.setLayout(qh)
        #显示窗口
        self.show()
    #控制窗口显示在屏幕中心的方法    
    def center(self):
        #获得窗口
        qr = self.frameGeometry()
        #获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        #显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    #重写关闭窗口事件
    def closeEvent(self, event): 
        #显示一个消息框，是或者不是
        reply = QMessageBox.question(self, '警告',"你确定要退出吗?", 
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No)#默认按钮
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()   
def main():
    #界面实例
    app=QApplication([])
    #QWidget部件是pyqt5所有用户界面对象的基类。他为QWidget提供默认构造函数。默认构造函数没有父类。
    w=Window()
    #点击关闭窗口才关闭窗口
    sys.exit(app.exec_())
if __name__ == "__main__":
    main()