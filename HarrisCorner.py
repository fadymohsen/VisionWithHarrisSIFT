import numpy as np
import cv2
import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel
import time





class HarrisCornerDetection():
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.threshold_ratio = 0.01  # Initialize threshold ratio
        self.ui.pushButton.clicked.connect(self.detect_corners)
        self.ui.horizontalSlider_6.valueChanged.connect(self.update_label_threshold)
        self.ui.horizontalSlider_6.setMinimum(0)
        self.ui.horizontalSlider_6.setMaximum(10)
        self.ui.horizontalSlider_6.setSingleStep(1)
        # Initialize scaled threshold ratio
        self.scaled_threshold_ratio = self.ui.horizontalSlider_6.value() / 1000.0
        # Check for label_threshold and update it
        if hasattr(self.ui, 'label_threshold'):
            self.threshold_label = self.ui.label_threshold
            self.update_label_threshold(self.ui.horizontalSlider_6.value())  # Initialize label text
        else:
            print("Label 'label_threshold' not found. Check your UI design.")
        self.original_image = None


    def update_label_threshold(self, value):
        # Update the text of the threshold label with current slider value and store scaled value
        self.scaled_threshold_ratio = value / 100.0
        if hasattr(self.ui, 'label_threshold'):
            self.ui.label_threshold.setText(f"{self.scaled_threshold_ratio:.2f}")
        else:
            print("Label 'label_threshold' not found.")


    def upload_image(self):
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


    def detect_corners(self):
        if self.original_image is None:
            print("No image loaded.")
            return  # Exit the function if no image has been loaded
        # Start the timer
        start_time = time.time()
        # Convert image to float32
        image = np.float32(self.original_image)
        # Calculate derivatives (image gradients) using Sobel operator
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        # Calculate products of derivatives
        IxIx = pow(sobel_x, 2)
        IxIy = sobel_x * sobel_y
        IyIy = pow(sobel_y, 2)
        # Apply Gaussian filter (3*3 kernel size) to the products of derivatives
        sum_IxIx = cv2.GaussianBlur(IxIx, (3,3), 0)  
        sum_IyIy = cv2.GaussianBlur(IyIy, (3,3), 0)    
        sum_IxIy = cv2.GaussianBlur(IxIy, (3,3), 0) 
        # Harris Corner Response Calculation
        k = 0.04   # commonly used value for k
        det_H = (sum_IxIx * sum_IyIy) - pow(sum_IxIy, 2)  # eigenValues multiplication
        trace_H = sum_IxIx + sum_IyIy                     # eigenValues summation
        harris_response = det_H - k * pow(trace_H, 2)
        # Thresholding
        threshold = self.scaled_threshold_ratio * np.max(harris_response)
        # Non-maximum suppression
        neighborhood_size = 1
        harris_response_max = cv2.dilate(harris_response, np.ones((neighborhood_size, neighborhood_size)))
        corner_mask = (harris_response == harris_response_max) & (harris_response > threshold)
        # Highlight corners wit red spots
        corners_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        corners_image[corner_mask] = [255, 0, 0]  
        # Display the image with the detected corners
        self.ui.graphicsLayoutWidget_afterHarris.clear()
        corners_img_item = pg.ImageItem(corners_image)
        corners_view = self.ui.graphicsLayoutWidget_afterHarris.addViewBox()
        corners_view.addItem(corners_img_item)
        # Stop the timer
        end_time = time.time()
        self.computation_time = end_time - start_time
        if hasattr(self.ui, 'label_computationalTime'):
            self.ui.label_computationalTime.setText(f"{self.computation_time:.4f} sec")
        else:
            print("Label 'label_computationalTime' not found. Check your UI design.")