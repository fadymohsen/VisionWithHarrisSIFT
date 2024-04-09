import numpy as np
import cv2
import pyqtgraph as pg
import time

class HarrisCornerDetection():
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget


    def UploadImage(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            self.ui.graphicsLayoutWidget_beforeHarris.clear()
            original_img_item = pg.ImageItem(imageArray)
            original_view = self.ui.graphicsLayoutWidget_beforeHarris.addViewBox()
            original_view.addItem(original_img_item)
            self.original_image = imageArray

