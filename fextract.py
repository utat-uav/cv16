import cv2
import numpy

# HSV testing

# Using saturation channel
ACTIVE_CHANNEL = 1

def main():
    # import image from file
    imgin = cv2.imread("im0216.jpg", cv2.IMREAD_COLOR)
    hsv_imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)

    hsv_chans = cv2.split(hsv_imgin); # split image into HSV channels

    # output image
    for i, hsvim in enumerate(hsv_chans):
        hsv_blur = cv2.blur(hsvim, (10, 10))  # blur image
        cv2.imwrite("hsv" + str(i) + ".jpg", hsv_blur)
        cv2.imwrite("hsv_blur" + str(i) + ".jpg", hsv_blur)
     
if __name__ == "__main__":
    main()
