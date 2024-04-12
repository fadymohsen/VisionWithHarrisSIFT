import numpy as np
import cv2
import cmath
import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel
import time





# img = cv2.imread('images\\free-printable-chess-board1.jpg')
# cv2.imshow('img',img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# start_time = time.time()
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
# img[dst > 0.01 * dst.max()] = [0, 0, 255]
# end_time = time.time()
# execution_time = end_time - start_time
# print("Computation Time in Harris corner using OpenCV:", execution_time, "seconds") # approximately equal 0.0034122467041015625 seconds
# cv2.imshow('dst', img)
# if cv2.waitKey(0) == 27:
# cv2.destroyAllWindows()





class HarrisCornerDetection():
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        # Initialize threshold ratio 1/100
        self.threshold_ratio = 0.01  
        self.ui.pushButton.clicked.connect(self.detect_corners)
        self.ui.horizontalSlider_6.valueChanged.connect(self.update_threshold)
        # Set the minimum value for the slider
        self.ui.horizontalSlider_6.setMinimum(0)
        #  Set the maximum value for the slider
        self.ui.horizontalSlider_6.setMaximum(500)
        # Set the step for the slider
        self.ui.horizontalSlider_6.setSingleStep(1)  
         # Create a QLabel to display the current threshold value
        self.threshold_label = QLabel()
        # Add the label to the layout
        self.ui.horizontalLayout_23.addWidget(self.threshold_label)
        # Update the label initially
        self.update_threshold_label(self.ui.horizontalSlider_6.value())


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


    def update_threshold(self, value):
        # Update the threshold ratio when the spinbox value changes
        self.threshold_ratio = value / 1000.0  
        # Trigger corner detection whenever the threshold changes
        self.detect_corners()
        self.update_threshold_label(value)


    def update_threshold_label(self, value):
        # Update the text of the threshold label
        self.threshold_label.setText(f": {value}")



    def detect_corners(self):
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
        threshold = self.threshold_ratio * np.max(harris_response)
        # print(np.max(harris_response))
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
        # Calculate the computation time only if it's the first time
        if not hasattr(self, 'computation_time'):
            self.computation_time = end_time - start_time
            # print("Computation Time in Harris corner implemented from scratch:", self.computation_time, "seconds")
            # Clear the QTextEdit before appending the new computation time
            self.ui.textEdit.clear()
            # Display computation time on QTextEdit
            self.ui.textEdit.append("{:.4f} seconds".format(self.computation_time))