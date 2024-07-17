import numpy as np
import cv2

class SkinDetector(object):
    # Initialize an object using an image. Convert the image to an HSV Image
    def __init__(self, img):
        self.imgMask = []
        self.output = []
        self.image = img

        self.HSVImg = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.binaryMask = self.HSVImg
    
    # Find skin mask by calling color segmentation and region segmentation methods
    def findSkin(self):
        self.colorSegmentation()
        self.regionSegmentation()

    # Color segmentation method returns all the pixels contained between a lower and upper range
    # These values were found by empirical experimentation and are giving good results to almost all the pictures
    def colorSegmentation(self):
        lowerHSV = np.array([0, 40, 0], dtype="uint8")
        upperHSV = np.array([25, 210, 240], dtype="uint8")

        maskHSV = cv2.inRange(self.HSVImg, lowerHSV, upperHSV)
        self.binaryMask = maskHSV
        
    # Region segmentation first erodes the mask from the color segmentation to find the sure foreground
    # Then the dilation of the mask gives back what is the sure background
    # The marker will be the sum of the two images so we can find the region that is unknown
    # By using the cv2.watershed method, a full mask containing all the required information is returned
    def regionSegmentation(self):
        fgImg = cv2.erode(self.binaryMask, None, iterations=3)  # Remove noise
        dilatedMask = cv2.dilate(self.binaryMask, None, iterations=3)
        ret, bgImg = cv2.threshold(dilatedMask, 1, 128, cv2.THRESH_BINARY) 

        marker = cv2.add(fgImg, bgImg)

        # Convert to uint32 for watershedding
        marker32 = np.int32(marker)
        cv2.watershed(self.image, marker32)

        # Convert back to uint8 for processing
        m = cv2.convertScaleAbs(marker32)  
        ret, mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.imgMask = mask
        self.output = cv2.bitwise_and(self.image, self.image, mask=mask)

    # Return the final skin mask
    def getMask(self):
        return self.imgMask
