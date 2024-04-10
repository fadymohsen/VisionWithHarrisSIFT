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


    # def Browse(self):
    #     options = QFileDialog.Options()
    #     fileName, _ = QFileDialog.getOpenFileName(self.ui, "Select Image", "",
    #                                             "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
    #                                             options=options)
    #     if fileName:  # Check if a file was selected
    #         imageArray = cv2.imread(fileName)  

    #         if imageArray.ndim == 3:  
    #             imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)

    #         # Rotate the image
    #         imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            
    #         # Clear existing image
    #         self.ui.image_noiseBeforeEditing.clear()
            
    #         # Display the original image
    #         self.original_img_item = pg.ImageItem(imageArray)
    #         original_view = self.ui.image_noiseBeforeEditing.addViewBox()
    #         original_view.addItem(self.original_img_item)
    #         self.original_image = imageArray
