from PyQt5.QtWidgets import QMainWindow, QApplication, QTabWidget
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt







class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        loadUi('MainWindow.ui', self.tab_widget)
        self.full_screen = False




    def keyPressEvent(self, event):
        if event.key() == 16777216:         # Integer value for Qt.Key_Escape
            if self.isFullScreen():
                self.showNormal()           # Show in normal mode
            else:
                self.showFullScreen()       # Show in full screen
        else:
            super().keyPressEvent(event)









if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
