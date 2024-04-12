from PyQt5.QtWidgets import QMainWindow, QApplication, QTabWidget
from PyQt5.QtWidgets import QApplication, QTabWidget, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from Matching_image import Template_Matching
from SIFT import SIFTCornerDetection
from HarrisCorner import HarrisCornerDetection
import cv2



class MainWindow(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        loadUi(ui_file, self)
        self.full_screen = False
        self.showFullScreen()
        self.pushButton_browseImage.clicked.connect(self.browse_image)
        self.template_matching = Template_Matching(self)
        self.addSIFT = SIFTCornerDetection(self)
        self.Harris = HarrisCornerDetection(self)
        self.template_matching.handle_buttons()


    def browse_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        if file_name:
            self.selected_image_path = file_name
            self.display_image_on_graphics_layout(file_name)
            self.Harris.upload_image()
            # self.addSIFT.SIFTDetector()
            

    def display_image_on_graphics_layout(self, image_path):
        image_data = cv2.imread(image_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = np.rot90(image_data, -1)
        # Clear the previous image if any
        self.graphicsLayoutWidget_displayImagesMain.clear()
        # Create a PlotItem or ViewBox
        view_box = self.graphicsLayoutWidget_displayImagesMain.addViewBox()
        # Create an ImageItem and add it to the ViewBox
        image_item = pg.ImageItem(image_data)
        view_box.addItem(image_item)
        # Optional: Adjust the view to fit the image
        view_box.autoRange()

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
        if event.key() == 16777216:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)









if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow("MainWindow.ui")
    window.show()
    app.exec_()
