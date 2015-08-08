import cv2
import numpy

def main():
    # Get both files for image processing
    imgin = []
    imgin_blur = []
    for i in range(0,3):
        imgin_blur.append(cv2.imread("hsv_blur" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE))
        imgin.append(cv2.imread("hsv" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE))

    myMSER = cv2.MSER_create() 
    
    kpts = []
    dkpsout = []
    for i, im in enumerate(imgin):
        local_kpt = myMSER.detect(im)
        kpts.append([local_kpt])
        if (local_kpt):
            local_dpksout = cv2.drawKeypoints(im, local_kpt, im) 
            dkpsout.append(local_dpksout)
            cv2.imwrite('dkpsout' + str(i) + '.jpg', local_dpksout)

if __name__ == "__main__":
    main()
