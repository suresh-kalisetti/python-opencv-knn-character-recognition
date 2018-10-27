# GenData.py

import sys
import numpy as np
import cv2
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    testfolders = os.listdir(r"D:\Python\Hackathon\LPR\TestData")

    intClassifications = []
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    sampleSize = 100

    for testfolder in testfolders:
        label = str(testfolder)
        count = 0
        samples = os.listdir(r"D:\Python\Hackathon\LPR\TestData\\" + label +"\\")
        for sample in samples:
            if count < sampleSize:
                imgTrainingData = cv2.imread(r"D:\Python\Hackathon\LPR\TestData\\" + label + "\\" + sample)
                imgGray = cv2.cvtColor(imgTrainingData, cv2.COLOR_BGR2GRAY)
                imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
                imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                        255,                                  # make pixels that pass the threshold full white
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                        cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                        11,                                   # size of a pixel neighborhood used to calculate threshold value
                                        2)                                    # constant subtracted from the mean or weighted mean
                imgThreshCopy = imgThresh.copy()
                imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                for npaContour in npaContours:
                    if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
                        [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
                        cv2.rectangle(imgTrainingData,
                                        (intX, intY),
                                        (intX+intW,intY+intH),
                                        (0, 0, 255),
                                        2)
                        imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
                        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                        
                        #cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference

                        intClassifications.append(ord(label))
                        npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                        npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
                # end for
                count = count + 1
            # end if
            else:
                break
        # end for
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




