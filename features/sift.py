import numpy as np
import cv2
import pyqtgraph as pg
import time
from numpy.linalg import det, lstsq, norm
from functools import cmp_to_key
from typing import List, Tuple, Optional



class SIFTCornerDetection:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.sigma = 1.6  # Example, adjust based on your Gaussian blurring

    
    def uploadImageSIFT(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            self.ui.graphicsLayoutWidget_beforeSIFT.clear()
            original_img_item = pg.ImageItem(imageArray)
            original_view = self.ui.graphicsLayoutWidget_beforeSIFT.addViewBox()
            original_view.addItem(original_img_item)
            self.original_image = imageArray.astype('float32')


    def uploadImageForMatching(self, path):
        imageArray = cv2.imread(path)
        if imageArray.ndim == 3:
            imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
        imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
        self.ui.graphicsLayoutWidget_beforeSIFT.clear()
        original_img_item = pg.ImageItem(imageArray)
        original_view = self.ui.graphicsLayoutWidget_beforeSIFT.addViewBox()
        original_view.addItem(original_img_item)
        self.original_image = imageArray.astype('float32')


    def SIFTDetector(self):
            startTime = time.time()
            # Creating a Base Image to start by over sampling by doubling the image in size in both direction
            initialImage = cv2.resize(self.original_image, (0, 0), fx=2, fy=2, interpolation= cv2.INTER_LINEAR)
            # In order to start the image with sigma = 1.6 we blur the over sampling image by sigma diff
            # Assuming original image has blur of 0.5 
            startSigma = 1.6
            sigma_diff = np.sqrt(max((startSigma ** 2) - ((2 * 0.5) ** 2), 0.01))
            baseImage = cv2.GaussianBlur(initialImage, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff) 
            #Calculating the number of possible octaves that will result of minimum image of 3 * 3
            baseImageShape = baseImage.shape
            numOfOctaves = int(round(np.log(min(baseImageShape)) / np.log(2) - 1))
            print(f"number of octaves = ", numOfOctaves)
            #Create a list of the amount of blur for each image in a particular layer
            numOfIntervals = 3
            numOfImagesPerOctave  = 6
            k = 2 ** (1. / numOfIntervals)
            gaussianKernels = np.zeros(numOfImagesPerOctave)  
            gaussianKernels[0] = startSigma #Start sigma = 1.6
            for image_index in range(1, numOfImagesPerOctave ):
                sigma_previous = (k ** (image_index - 1)) * 1.6
                sigma_total = k * sigma_previous
                gaussianKernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
            #Finding the Gaussian Images for each octave to prepare for DOG
            gaussianImages = []
            for octaveIndex in range(numOfOctaves):
                gaussian_images_in_octave = []
                gaussian_images_in_octave.append(baseImage)  # first image in octave already has the correct blur
                for gaussianKernel in gaussianKernels[1:]:
                    image = cv2.GaussianBlur(baseImage, (0, 0), sigmaX=gaussianKernel, sigmaY=gaussianKernel)
                    gaussian_images_in_octave.append(image)
                gaussianImages.append(gaussian_images_in_octave)
                octave_base = gaussian_images_in_octave[-3]
                # Down Sampling the images in each octave for half
                image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2),
                                                  int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
            gaussianImages = np.array(gaussianImages)
            #Now generating the DOG Images
            DOGImages = []
            for gaussian_images_in_octave in gaussianImages:
                dog_images_in_octave = []
                #Subtracting adjacent images in the gaussian images
                for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                    dog_images_in_octave.append(np.subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
                DOGImages.append(dog_images_in_octave)
            DOGImages =  np.array(DOGImages)
            #Now we Find pixel positions of all scale-space extrema in all octaves
            #Some constants from openCV implementation found in papers
            contrastThreshold=0.04
            threshold = np.floor(0.5 * contrastThreshold / numOfIntervals * 255) 
            self. keyPoints = []

            for octaveIndex, DOGImageInOctave in enumerate(DOGImages):
                for imageIndex, (firstImage, secondImage, thirdImage) in enumerate(zip(DOGImageInOctave,
                                                                                        DOGImageInOctave[1:],
                                                                                          DOGImageInOctave[2:])):
                    #We exclude a 5px border to avoid edge effects
                    imageBorderWidthExcluded = 5
                    for i in range(imageBorderWidthExcluded, firstImage.shape[0] - imageBorderWidthExcluded):
                        for j in range(imageBorderWidthExcluded, firstImage.shape[1] - imageBorderWidthExcluded):
                            #extracting 3x3 subMatrices centered at pixel (i, j)
                            firstSubMatrix = firstImage[i-1:i+2, j-1:j+2]
                            secondSubMatrix = secondImage[i-1:i+2, j-1:j+2]
                            thirdSubMatrix = thirdImage[i-1:i+2, j-1:j+2]
                            if (self.isPixelMinimaOrMaxima(firstSubMatrix, secondSubMatrix, thirdSubMatrix, threshold)):
                                localization_result = self.localizeExtremum(i, j, imageIndex + 1, octaveIndex, numOfIntervals, DOGImageInOctave, startSigma, contrastThreshold, imageBorderWidthExcluded)
                                if localization_result is not None:
                                    keyPoint, localized_image_index = localization_result
                                    self.keyPoints.append(keyPoint)



            self.keypoints = self.remove_duplicate_keypoints(self.keyPoints)
            self.keypoints = self.convert_keypoints_to_input_image_size(self.keyPoints)
            self.descriptors = self.generate_descriptors(self.keyPoints, gaussianImages)
            

                                  
            endTime = time.time()
            totalTime = endTime - startTime
            self.ui.label_SIFTcomputationTime.setText(str(totalTime))
            # Example final call to display image
            # Display the processed image with keypoints
            # self.displayFinalImage(self.original_image)
            self.displayFinalImage( self.original_image, self.keyPoints)


            
    

    def isPixelMinimaOrMaxima(self, firstSubMatrix, secondSubMatrix, thirdSubMatrix, threshold):
        #Finding if a centred pixel is local minima or maxima
        centredPixel = secondSubMatrix[1, 1]
        if abs(centredPixel)>threshold:
            #Checking if it's local maxima
            if centredPixel > 0:
                if all(centredPixel >= firstSubMatrix.flatten()) and \
                    all(centredPixel >= thirdSubMatrix.flatten()) and \
                all(centredPixel >= secondSubMatrix[0,: ].flatten()) and\
                    all(centredPixel >= secondSubMatrix[2,: ].flatten()) and \
                all(centredPixel >= secondSubMatrix[1,0].flatten()) and \
                    all(centredPixel >= secondSubMatrix[1,2].flatten()):
                    return True
            #Checking if it's local minima
            elif centredPixel < 0:
                if all(centredPixel <= firstSubMatrix.flatten()) and \
                    all(centredPixel <= thirdSubMatrix.flatten()) and \
                all(centredPixel <= secondSubMatrix[0,: ].flatten()) and\
                    all(centredPixel <= secondSubMatrix[2,: ].flatten()) and \
                all(centredPixel <= secondSubMatrix[1,0].flatten()) and \
                    all(centredPixel <= secondSubMatrix[1,2].flatten()):
                    return True
        return False

    def localizeExtremum(self, i, j, image_index, octaveIndex, numOfIntervals, DOGImageInOctave, sigma, contrastThreshold, imageBorderWidthExcluded, eigenvalueRatio=10, num_attempts_until_convergence=5):
        #Iteratively refine pixel positions of scale-space extrema 
        extremum_is_outside_image = False
        imageShape = DOGImageInOctave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            firstImage, secondImage, thirdImage = DOGImageInOctave[image_index-1:image_index+2]
            pixel_cube = np.stack([firstImage[i-1:i+2, j-1:j+2],
                                secondImage[i-1:i+2, j-1:j+2],
                                thirdImage[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.computeGradientAtCenterPixel(pixel_cube)
            hessian = self.computeHessianAtCenterPixel(pixel_cube)
            extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < imageBorderWidthExcluded or i >= imageShape[0] - imageBorderWidthExcluded or j < imageBorderWidthExcluded or j >= imageShape[1] - imageBorderWidthExcluded or image_index < 1 or image_index > numOfIntervals:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * numOfIntervals >= contrastThreshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalueRatio * (xy_hessian_trace ** 2) < ((eigenvalueRatio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV keyPoint object
                keyPoint = cv2.KeyPoint()
                keyPoint.pt = ((j + extremum_update[0]) * (2 ** octaveIndex), (i + extremum_update[1]) * (2 ** octaveIndex))
                keyPoint.octave = octaveIndex + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keyPoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(numOfIntervals))) * (2 ** (octaveIndex + 1))  # octaveIndex + 1 because the input image was doubled
                keyPoint.response = abs(functionValueAtUpdatedExtremum)
                return keyPoint, image_index
        return None

    def computeGradientAtCenterPixel(self, pixelArray):
        #Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula , where h is the step size
        # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
        dx = 0.5 * (pixelArray[1, 1, 2] - pixelArray[1, 1, 0])
        dy = 0.5 * (pixelArray[1, 2, 1] - pixelArray[1, 0, 1])
        ds = 0.5 * (pixelArray[2, 1, 1] - pixelArray[0, 1, 1])
        return np.array([dx, dy, ds])

    def computeHessianAtCenterPixel(self, pixelArray):
        center_pixel_value = pixelArray[1, 1, 1]
        dxx = pixelArray[1, 1, 2] - 2 * center_pixel_value + pixelArray[1, 1, 0]
        dyy = pixelArray[1, 2, 1] - 2 * center_pixel_value + pixelArray[1, 0, 1]
        dss = pixelArray[2, 1, 1] - 2 * center_pixel_value + pixelArray[0, 1, 1]
        dxy = 0.25 * (pixelArray[1, 2, 2] - pixelArray[1, 2, 0] - pixelArray[1, 0, 2] + pixelArray[1, 0, 0])
        dxs = 0.25 * (pixelArray[2, 1, 2] - pixelArray[2, 1, 0] - pixelArray[0, 1, 2] + pixelArray[0, 1, 0])
        dys = 0.25 * (pixelArray[2, 2, 1] - pixelArray[2, 0, 1] - pixelArray[0, 2, 1] + pixelArray[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                            [dxy, dyy, dys],
                            [dxs, dys, dss]])
    
    def displayFinalImage(self, image, keyPoints):
        self.ui.graphicsLayoutWidget_afterSIFT.clear()
        image = np.copy(image)  # Ensure you're working on a copy of the image to avoid modifying the original
        
        # Convert grayscale to a 3-channel image if necessary
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw red circles at each keypoint location
        for keyPoint in keyPoints:
            # Extract the x and y coordinates of the keypoint
            x, y = int(keyPoint.pt[0]), int(keyPoint.pt[1])
            # Draw a circle at each keypoint
            cv2.circle(image, (y, x), radius=1, color=(0, 0, 255), thickness=-1)  # Red circle with radius 3

        # Convert the image from BGR to RGB for display in PyQtGraph
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create an ImageItem and add it to the graphics layout
        original_img_item = pg.ImageItem(image_display)
        

        original_view = self.ui.graphicsLayoutWidget_afterSIFT.addViewBox()
        original_view.addItem(original_img_item)
        print ("hi1")
        
        
        # # Add red dots for keypoints
        # keyPoints_x = [kp[1] for kp in keyPoints_tuples]
        # keyPoints_y = [kp[0] for kp in keyPoints_tuples]
        # keyPoints_size = 5  # Adjust the size of the red dots if needed
        # keyPoints_pen = pg.mkPen(None)  # No outline for the red dots
        # keyPoints_brush = pg.mkBrush('r')  # Red fill color for the red dots
        # keyPoints_item = pg.ScatterPlotItem(size=keyPoints_size, pen=keyPoints_pen, brush=keyPoints_brush)
        # keyPoints_item.setData(pos=list(zip(keyPoints_x, keyPoints_y)))
        # print('hi2')
        # original_view.addItem(keyPoints_item)
        # print('hi3')


        

    """
    def computeKeypointOrientations(self, keypoints, gaussianImage):
        # Constants for orientation computation
        radius = 3 * self.sigma
        num_bins = 36
        hist = np.zeros(num_bins)

        for keypoint in keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            weight_window = gaussianImage[y - radius:y + radius + 1, x - radius:x + radius + 1]
            exp_scale = -1 / (2.0 * (0.5 * (2 * radius) * (0.5 * (2 * radius))))

            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if (0 <= y + i < gaussianImage.shape[0]) and (0 <= x + j < gaussianImage.shape[1]):
                        dx = gaussianImage[y + i, x + j + 1] - gaussianImage[y + i, x + j - 1]
                        dy = gaussianImage[y + i + 1, x + j] - gaussianImage[y + i - 1, x + j]
                        magnitude = np.sqrt(dx * dx + dy * dy)
                        orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(exp_scale * (i * i + j * j)) * magnitude
                        hist[int(orientation // (360 // num_bins))] += weight

            dominant_orientation = np.argmax(hist) * (360.0 / num_bins)
            keypoint.angle = dominant_orientation  # Adding orientation to the keypoint



    def createDescriptors(self, keypoints, gaussianImage):
        descriptor_length = 128  # 16x16 neighborhood, divided into 4x4 blocks, each contributes 8 bins histogram
        self.keypoint_descriptors = []

        for keypoint in keypoints:
            descriptor = np.zeros(descriptor_length)
            angle = 360 - keypoint.angle
            angle_rad = np.deg2rad(angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            # 16x16 neighborhood around the keypoint
            for i in range(-8, 8):
                for j in range(-8, 8):
                    # Rotate point by keypoint orientation
                    x_rotated = int(keypoint.pt[0] + (j * cos_angle - i * sin_angle))
                    y_rotated = int(keypoint.pt[1] + (j * sin_angle + i * cos_angle))
                    if 0 <= x_rotated < gaussianImage.shape[1] and 0 <= y_rotated < gaussianImage.shape[0]:
                        dx = gaussianImage[y_rotated, x_rotated + 1] - gaussianImage[y_rotated, x_rotated - 1]
                        dy = gaussianImage[y_rotated + 1, x_rotated] - gaussianImage[y_rotated - 1, x_rotated]
                        magnitude = np.sqrt(dx * dx + dy * dy)
                        orientation = (np.rad2deg(np.arctan2(dy, dx)) - angle) % 360
                        bin_idx = int((orientation / 360) * descriptor_length)
                        descriptor[bin_idx] += magnitude
            self.keypoint_descriptors.append(descriptor / np.linalg.norm(descriptor))  # Normalize the descriptor    



    def get_keypoints_descriptors(self):
        self.SIFTDetector()
        return self.keyPoints, self.keypoint_descriptors
    """




    def computeKeypointsWithOrientations(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36,          peak_ratio=0.8, scale_factor=1.5):
        """Compute orientations for each keypoint
        """
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                # Quadratic peak interpolation
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < np.float_tolerance:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    def compare_keypoints(self, keypoint1: cv2.KeyPoint, keypoint2: cv2.KeyPoint) -> int:
        """
        Comparator function for keypoints to sort and remove duplicates.

        Parameters:
            keypoint1 (cv2.KeyPoint): First keypoint.
            keypoint2 (cv2.KeyPoint): Second keypoint.

        Returns:
            int: Comparison result for sorting.
        """
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def remove_duplicate_keypoints(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        Sort keypoints and remove duplicates to ensure unique keypoints.

        Parameters:
            keypoints (List[cv2.KeyPoint]): List of keypoints to filter.

        Returns:
            List[cv2.KeyPoint]: List of unique keypoints.
        """
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compare_keypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints

    def convert_keypoints_to_input_image_size(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        Adjust keypoint coordinates and size back to the dimensions of the input image.

        Parameters:
            keypoints (List[cv2.KeyPoint]): List of keypoints detected in the processed image.

        Returns:
            List[cv2.KeyPoint]: List of keypoints adjusted to the input image size.
        """
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def unpack_octave(self, keyPoint: cv2.KeyPoint) -> Tuple[int, int, float]:
        """
        Decompose the octave information from a keypoint.

        Parameters:
            keypoint (cv2.KeyPoint): The keypoint to unpack.

        Returns:
            Tuple[int, int, float]: The octave, layer, and scale from the keypoint.
        """
        octave = self.keyPoints.octave & 255
        layer = (self.keyPoints.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale

    def generate_descriptors(self, keypoints: List[cv2.KeyPoint], gaussianImages: np.ndarray, window_width: int = 4, num_bins: int = 8, scale_multiplier: float = 3, descriptor_max_value: float = 0.2) -> np.ndarray:
        """
        Generate descriptors for each keypoint based on their orientation histograms.

        Parameters:
            keypoints (List[cv2.KeyPoint]): List of keypoints.
            gaussianImages (np.ndarray): The Gaussian images corresponding to each octave and layer.
            window_width (int): The width of the window used to consider the neighborhood of each keypoint.
            num_bins (int): Number of bins for the orientation histogram.
            scale_multiplier (float): Multiplier for scaling the keypoint size to determine the descriptor window.
            descriptor_max_value (float): Maximum value for clipping the descriptor elements.

        Returns:
            np.ndarray: Array of keypoint descriptors.
        """
        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self.unpack_octave(keypoint)
            gaussian_image = gaussianImages[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if -1 < row_bin < window_width and -1 < col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_tolerance)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')



    def get_keypoints_descriptors(self):
        self.SIFTDetector()
        print(f"lenght:{len(self.descriptors)}")
        return self.keypoints, self.descriptors 