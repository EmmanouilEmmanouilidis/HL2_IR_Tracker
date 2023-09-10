from pickle import FALSE
import cv2
import numpy as np


class Calibration:
    def __init__(self):
        ir_img = cv2.imread('ir_image_2.tiff', cv2.IMREAD_GRAYSCALE)
        data = np.asarray(ir_img)

        _, th = cv2.threshold(data, 15, 255, cv2.THRESH_BINARY)

        width = int(data.shape[1])
        height = int(data.shape[0])
        dims = (width//2, height//2)
        resized = cv2.resize(th, dims, interpolation=cv2.INTER_LINEAR)

        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 5
        #params.maxThreshold = 255
        params.minDistBetweenBlobs = 0

        # Filter by Area.
        params.filterByArea = False
        params.minArea = 1

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.7

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.9

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.5

        # |Filter by Color
        params.filterByColor = True
        params.blobColor = 255

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(resized)
        for kp in keypoints:
            print(kp.pt)
        #print(resized.shape)
        im_with_keypoints = cv2.drawKeypoints(resized, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("keypts", im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_blobs(self):
        pass


if __name__ == "__main__":
    calib = Calibration()
    calib.detect_blobs()