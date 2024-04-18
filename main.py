from PyQt5.QtWidgets import QApplication, QTabWidget
from PyQt5.QtWidgets import QApplication, QTabWidget, QFileDialog
from PyQt5.uic import loadUi
import numpy as np
import pyqtgraph as pg
import cv2
from Features.MatchingImage import TemplateMatching
from Features.sift import SIFT
from Features.HarrisCorner import HarrisCornerDetection



class MainWindow(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        loadUi(ui_file, self)
        self.full_screen = False
        self.showFullScreen()
        self.pushButton_browseImage.clicked.connect(self.browse_image)
        self.btn_SIFT.clicked.connect(self.apply_sift)
        self.template_matching = TemplateMatching(self)
        self.Harris = HarrisCornerDetection(self)
        self.template_matching.handle_buttons()



    def browse_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        if file_name:
            self.selected_image_path = file_name
            image_data = cv2.imread(file_name)
            self.display_image(self.graphicsLayoutWidget_displayImagesMain, image_data)
            self.display_image(self.graphicsLayoutWidget_beforeSIFT, image_data)

            self.Harris.upload_image()
            self.addSIFT = SIFT(image_data,self)

    def apply_sift(self):
        self.addSIFT.sift()         


    def display_image(self, graph_name, image_data):
        print("Type of image_data:", type(image_data))
        if isinstance(image_data, np.ndarray):
            print("Shape of image_data:", image_data.shape)
        else:
            print("image_data is not a numpy array")

        # Ensure the image data is an array and has at least two dimensions
        if not isinstance(image_data, np.ndarray) or image_data.ndim < 2:
            print("Invalid image data provided")
            return  # Exit the function if image data is not valid

        image_data = np.rot90(image_data, -1)
        graph_name.clear()
        view_box = graph_name.addViewBox()
        image_item = pg.ImageItem(image_data)
        view_box.addItem(image_item)
        view_box.autoRange()



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
