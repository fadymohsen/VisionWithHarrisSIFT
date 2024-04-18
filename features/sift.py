from functools import cmp_to_key
import numpy as np
import time
import cv2
import pyqtgraph as pg





class SIFT:
    def __init__(self, original_image,tab_widget):
        self.ui = tab_widget
        self.start_time = time.time()
        # original_image = cv2.imread(file_path,0)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # convert image to gray scale
        self.original_image = original_image.astype('float32')
        




    def sift(self, sigma=1.6, no_of_levels=3, assumed_blur=0.5, imageBorderExecluded=5):
        # Creating the Scale Space
        image = cv2.resize(self.original_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
        numOfOctaves = int(np.round(np.log(min(image.shape)) / np.log(2) - 1))
        gaussianPyramid = self.createGaussianPyramidGivingNumOfOctaves(image, numOfOctaves, sigma, no_of_levels)
        DOGImages = self.createDoGPyramidFromGaussianPyramid(gaussianPyramid)

        # Extracting Keypoints and Descriptors
        keypoints = self.localizeKeypointsGivingGaussianPyramidAndDoG(gaussianPyramid, DOGImages, no_of_levels, sigma, imageBorderExecluded)
        keypoints = self.removeDuplicateKeypoints(keypoints)
        keypoints = self.convertKeypointsToInputImageSize(keypoints)
        self.displayImage(keypoints)
        descriptors = self.generateDescriptorsFromKeypoints(keypoints, gaussianPyramid)

        print(f"SIFT Computation time: {time.time() - self.start_time}")
        return keypoints, descriptors
    


    def displayImage(self,keypoints_list):
        image = np.copy(self.original_image)
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for keypoint in keypoints_list:
            y,x = int(np.round(keypoint.pt[0])), int(np.round(keypoint.pt[1]))
            image[x:x+3,y:y+3,0] = 255
            image[x:x+3,y:y+3,1] = 0
            image[x:x+3,y:y+3,2] = 0

        image= np.rot90(image, -1)
        # cv2.imwrite('keypoints2.png', image)
        self.ui.graphicsLayoutWidget_afterSIFT.clear()
        view_box = self.ui.graphicsLayoutWidget_afterSIFT.addViewBox()
        image_item = pg.ImageItem(image)
        view_box.addItem(image_item)
        view_box.autoRange()
    


    def calcSigmaValuesUsingNumImagesPerOctave(self, sigma, no_of_levels):
        num_images_per_octave = no_of_levels + 3
        k = 2 ** (1. / no_of_levels)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = sigma

        for imageIndex in range(1, num_images_per_octave):
            sigma_previous = (k ** (imageIndex - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[imageIndex] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        
        return gaussian_kernels



    def createGaussianPyramidGivingNumOfOctaves(self, image, numOfOctaves, sigma, no_of_levels):
        gaussianPyramid = []
        sigmas = self.calcSigmaValuesUsingNumImagesPerOctave(sigma, no_of_levels)
        gaussianPyramid = []
        for octave in range(numOfOctaves):
            gaussian_levels = []
            for sigma in sigmas:
                if len(gaussian_levels) == 0:
                    gaussian_levels.append(image)
                else:
                    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                    gaussian_levels.append(image)
            gaussianPyramid.append(gaussian_levels)
            if octave < numOfOctaves - 1:
                image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        return np.array(gaussianPyramid, dtype=object)



    def createDoGPyramidFromGaussianPyramid(self, gaussianPyramid):
        DOGImages = []
        for gaussian_levels in gaussianPyramid:
            DOGImagesPerOctave = []
            for firstImage, secondImage in zip(gaussian_levels, gaussian_levels[1:]):
                DOGImagesPerOctave.append(np.subtract(secondImage, firstImage))
            DOGImages.append(DOGImagesPerOctave)
        return np.array(DOGImages, dtype=object)



    def localizeKeypointsGivingGaussianPyramidAndDoG(self, gaussianPyramid, DOGImages, no_of_levels, sigma, imageBorderExecluded, contrastThreshold=0.04):
        threshold = np.floor(0.5 * contrastThreshold / no_of_levels * 255)
        keypoints = []

        for octaveIndex, DOGImagesPerOctave in enumerate(DOGImages):
            for imageIndex, (firstImage, secondImage, thirdImage) in enumerate(zip(DOGImagesPerOctave, DOGImagesPerOctave[1:], DOGImagesPerOctave[2:])):
                for i in range(imageBorderExecluded, firstImage.shape[0] - imageBorderExecluded):
                    for j in range(imageBorderExecluded, firstImage.shape[1] - imageBorderExecluded):
                        firstSubimage = firstImage[i-1:i+2, j-1:j+2]
                        secondSubimage = secondImage[i-1:i+2, j-1:j+2]
                        thirdSubimage = thirdImage[i-1:i+2, j-1:j+2]
                        if self.is_extreme(firstSubimage, secondSubimage, thirdSubimage, threshold):
                            localizationResult = self.localizeExtremumViaQuadraticFit(i, j, imageIndex + 1, octaveIndex, no_of_levels, DOGImagesPerOctave, sigma, contrastThreshold, imageBorderExecluded)
                            if localizationResult is not None:
                                keypoint, localized_imageIndex = localizationResult
                                keypointsWithOrientations = self.computeKeypointsWithOrientations(keypoint, octaveIndex, gaussianPyramid[octaveIndex][localized_imageIndex])
                                for keypoint in keypointsWithOrientations:
                                    keypoints.append(keypoint)

        return keypoints



    def is_extreme(self, firstSubimage, secondSubimage, thirdSubimage, threshold):
        #checking if pixel is local minima or maxima
        centerPixelValue = secondSubimage[1, 1]

        if np.all(np.abs(centerPixelValue) <= threshold):
            return False
        
        if np.all(centerPixelValue > 0):
            return (
                np.all(centerPixelValue >= firstSubimage.max()) and
                np.all(centerPixelValue >= thirdSubimage.max()) and
                np.all(centerPixelValue >= secondSubimage.max())
            )
        else:
            return (
                np.all(centerPixelValue <= firstSubimage.min()) and
                np.all(centerPixelValue <= thirdSubimage.min()) and
                np.all(centerPixelValue <= secondSubimage.min())
            )



    def localizeExtremumViaQuadraticFit(self, i, j, imageIndex, octaveIndex, no_of_levels, DOGImagesPerOctave, sigma, contrastThreshold, imageBorderExecluded, eigenvalue_ratio=10, num_attempts_until_convergence=5):
        extremum_is_outside_image = False
        imageShape = DOGImagesPerOctave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            firstImage, secondImage, thirdImage = DOGImagesPerOctave[imageIndex-1:imageIndex+2]
            pixel_cube = np.stack([firstImage[i-1:i+2, j-1:j+2],
                                secondImage[i-1:i+2, j-1:j+2],
                                thirdImage[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.computeGradientAtCenterPixel(pixel_cube)
            hessian = self.computeHessianAtCenterPixel(pixel_cube)
            #solving the gradients and hessian matrix
            updateTowardsExtrema = -np.linalg.lstsq(hessian.reshape(-1, 3), gradient.reshape(-1, 1), rcond=None)[0]
            if np.abs(updateTowardsExtrema[0]) < 0.5 and np.abs(updateTowardsExtrema[1]) < 0.5 and np.abs(updateTowardsExtrema[2]) < 0.5:
                break
            j += int(np.round(updateTowardsExtrema[0]))
            i += int(np.round(updateTowardsExtrema[1]))
            imageIndex += int(np.round(updateTowardsExtrema[2]))
            if i < imageBorderExecluded or i >= imageShape[0] - imageBorderExecluded or j < imageBorderExecluded or j >= imageShape[1] - imageBorderExecluded or imageIndex < 1 or imageIndex > no_of_levels:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, updateTowardsExtrema)
        if np.all(np.abs(functionValueAtUpdatedExtremum) * no_of_levels >= contrastThreshold):
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian = hessian[:2, :2]
            if xy_hessian.shape[0] != xy_hessian.shape[1]:
                return None
            xy_hessian_det = np.linalg.det(xy_hessian)

            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((j + updateTowardsExtrema[0][0]) * (2 ** octaveIndex)), ((i + updateTowardsExtrema[1][0]) * (2 ** octaveIndex))
                keypoint.octave = octaveIndex + imageIndex * (2 ** 8) + int(np.round((updateTowardsExtrema[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = int(sigma * (2 ** ((imageIndex + updateTowardsExtrema[2]) / np.float32(no_of_levels))) * (2 ** (octaveIndex + 1)))
                keypoint.response = abs(functionValueAtUpdatedExtremum[0])
                return keypoint, imageIndex
        return None



    def computeGradientAtCenterPixel(self, pixelArray):
        dx = 0.5 * (pixelArray[1, 1, 2] - pixelArray[1, 1, 0])
        dy = 0.5 * (pixelArray[1, 2, 1] - pixelArray[1, 0, 1])
        ds = 0.5 * (pixelArray[2, 1, 1] - pixelArray[0, 1, 1])
        return np.array([dx, dy, ds])



    def computeHessianAtCenterPixel(self, pixelArray):
        centerPixelValue = pixelArray[1, 1, 1]
        dxx = pixelArray[1, 1, 2] - 2 * centerPixelValue + pixelArray[1, 1, 0]
        dyy = pixelArray[1, 2, 1] - 2 * centerPixelValue + pixelArray[1, 0, 1]
        dss = pixelArray[2, 1, 1] - 2 * centerPixelValue + pixelArray[0, 1, 1]
        dxy = 0.25 * (pixelArray[1, 2, 2] - pixelArray[1, 2, 0] - pixelArray[1, 0, 2] + pixelArray[1, 0, 0])
        dxs = 0.25 * (pixelArray[2, 1, 2] - pixelArray[2, 1, 0] - pixelArray[0, 1, 2] + pixelArray[0, 1, 0])
        dys = 0.25 * (pixelArray[2, 2, 1] - pixelArray[2, 0, 1] - pixelArray[0, 2, 1] + pixelArray[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])



    def computeKeypointsWithOrientations(self, keypoint, octaveIndex, gaussianImage, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
        keypointsWithOrientations = []
        imageShape = gaussianImage.shape
        scale = scale_factor * keypoint.size / np.float32(2 ** (octaveIndex + 1))
        radius = int(np.round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        originalHistogram = np.zeros(num_bins)
        histogramAfterSmoothing = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(np.round(keypoint.pt[1] / np.float32(2 ** octaveIndex))) + i
            if region_y > 0 and region_y < imageShape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(np.round(keypoint.pt[0] / np.float32(2 ** octaveIndex))) + j
                    if region_x > 0 and region_x < imageShape[1] - 1:
                        dx = gaussianImage[region_y, region_x + 1] - gaussianImage[region_y, region_x - 1]
                        dy = gaussianImage[region_y - 1, region_x] - gaussianImage[region_y + 1, region_x]
                        gradientMagnitude = np.sqrt(dx * dx + dy * dy)
                        gradientOrientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_index = int(np.round(gradientOrientation * num_bins / 360.))
                        originalHistogram[histogram_index % num_bins] += weight * gradientMagnitude

        for n in range(num_bins):
            histogramAfterSmoothing[n] = (6 * originalHistogram[n] + 4 * (originalHistogram[n - 1] + originalHistogram[(n + 1) % num_bins]) + originalHistogram[n - 2] + originalHistogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(histogramAfterSmoothing)
        orientation_peaks = np.where(np.logical_and(histogramAfterSmoothing > np.roll(histogramAfterSmoothing, 1), histogramAfterSmoothing > np.roll(histogramAfterSmoothing, -1)))[0]
        for peakIndex in orientation_peaks:
            peakValue = histogramAfterSmoothing[peakIndex]
            if peakValue >= peak_ratio * orientation_max:
                leftValue = histogramAfterSmoothing[(peakIndex - 1) % num_bins]
                rightValue = histogramAfterSmoothing[(peakIndex + 1) % num_bins]
                interpolated_peakIndex = (peakIndex + 0.5 * (leftValue - rightValue) / (leftValue - 2 * peakValue + rightValue)) % num_bins
                orientation = 360. - interpolated_peakIndex * 360. / num_bins
                if abs(orientation - 360.) < 1e-7:
                    orientation = 0
                new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypointsWithOrientations.append(new_keypoint)
        return keypointsWithOrientations



    def compareKeypoints(self, keypoint1, keypoint2):
        """Return True if keypoint1 is less than keypoint2
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



    def removeDuplicateKeypoints(self, keypoints):
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compareKeypoints))
        uniqueKeypoints = [keypoints[0]]

        for nextKeypoint in keypoints[1:]:
            lastUniqueKeypoints = uniqueKeypoints[-1]
            if lastUniqueKeypoints.pt[0] != nextKeypoint.pt[0] or \
            lastUniqueKeypoints.pt[1] != nextKeypoint.pt[1] or \
            lastUniqueKeypoints.size != nextKeypoint.size or \
            lastUniqueKeypoints.angle != nextKeypoint.angle:
                uniqueKeypoints.append(nextKeypoint)
        return uniqueKeypoints



    def convertKeypointsToInputImageSize(self, keypoints):
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints



    def unpackOctave(self, keypoint):
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale



    def generateDescriptorsFromKeypoints(self, keypoints, gaussianPyramid, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):

        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = self.unpackOctave(keypoint)
            gaussianImage = gaussianPyramid[octave + 1, layer]
            numOfRows, numOfColumns = gaussianImage.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            np.cos_angle = np.cos(np.deg2rad(angle))
            np.sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogramTensor = np.zeros((window_width + 2, window_width + 2, num_bins))

            histogramWidth = scale_multiplier * 0.5 * scale * keypoint.size
            halfWidth = int(np.round(histogramWidth * np.sqrt(2) * (window_width + 1) * 0.5))
            halfWidth = int(min(halfWidth, np.sqrt(numOfRows ** 2 + numOfColumns ** 2)))

            for row in range(-halfWidth, halfWidth + 1):
                for col in range(-halfWidth, halfWidth + 1):
                    row_rot = col * np.sin_angle + row * np.cos_angle
                    col_rot = col * np.cos_angle - row * np.sin_angle
                    row_bin = (row_rot / histogramWidth) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / histogramWidth) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(np.round(point[1] + row))
                        window_col = int(np.round(point[0] + col))
                        if window_row > 0 and window_row < numOfRows - 1 and window_col > 0 and window_col < numOfColumns - 1:
                            dx = gaussianImage[window_row, window_col + 1] - gaussianImage[window_row, window_col - 1]
                            dy = gaussianImage[window_row - 1, window_col] - gaussianImage[window_row + 1, window_col]
                            gradientMagnitude = np.sqrt(dx * dx + dy * dy)
                            gradientOrientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / histogramWidth) ** 2 + (col_rot / histogramWidth) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradientMagnitude)
                            orientation_bin_list.append((gradientOrientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                rowFraction, columnFraction, orientationFraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * rowFraction
                c0 = magnitude * (1 - rowFraction)
                c11 = c1 * columnFraction
                c10 = c1 * (1 - columnFraction)
                c01 = c0 * columnFraction
                c00 = c0 * (1 - columnFraction)
                c111 = c11 * orientationFraction
                c110 = c11 * (1 - orientationFraction)
                c101 = c10 * orientationFraction
                c100 = c10 * (1 - orientationFraction)
                c011 = c01 * orientationFraction
                c010 = c01 * (1 - orientationFraction)
                c001 = c00 * orientationFraction
                c000 = c00 * (1 - orientationFraction)

                histogramTensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogramTensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogramTensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogramTensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogramTensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogramTensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogramTensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogramTensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptorVector = histogramTensor[1:-1, 1:-1, :].flatten()
            threshold = np.linalg.norm(descriptorVector) * descriptor_max_value
            descriptorVector[descriptorVector > threshold] = threshold
            descriptorVector /= max(np.linalg.norm(descriptorVector), 1e-7)
            descriptorVector = np.round(512 * descriptorVector)
            descriptorVector[descriptorVector < 0] = 0
            descriptorVector[descriptorVector > 255] = 255
            descriptors.append(descriptorVector)
        # self.display_discriptors(descriptors)
        return np.array(descriptors, dtype='float32')
