from PyQt5.QtWidgets import QMainWindow, QApplication, QTabWidget
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from Matching_image import Template_Matching
from SIFT import SIFTCornerDetection




class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        loadUi('MainWindow.ui', self.tab_widget)
        self.full_screen = False
        self.template_matching = Template_Matching(self)
        self.addSIFT = SIFTCornerDetection(self)
        # self.template_matching.handle_buttons()   




    def display_image(self,graphics_widget,image_data):
        """Utility function to display an image in a given graphics layout widget."""
        if image_data is not None:
            graphics_widget.clear()
            image_data = np.rot90(image_data, -1)
            view_box = graphics_widget.addViewBox()
            image_item = pg.ImageItem(image_data)
            view_box.addItem(image_item)
        else:
            print("Image data is not available.")



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
