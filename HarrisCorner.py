import numpy as np
import cv2
import cmath
import pyqtgraph as pg
import time


img = cv2.imread('images\\free-printable-chess-board1.jpg')
cv2.imshow('img',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
start_time = time.time()
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
end_time = time.time()
execution_time = end_time - start_time
print("Computation Time in Harris corner using OpenCV:", execution_time, "seconds")
cv2.imshow('dst', img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()