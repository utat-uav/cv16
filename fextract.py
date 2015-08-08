import cv2
import numpy

# HSV testing

# Using saturation channel
ACTIVE_CHANNEL = 1

def main():
    # import image from file
    imgin = cv2.imread("im0216.jpg", cv2.IMREAD_COLOR)
    hsv_imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)

    hsv_active = cv2.split(hsv_imgin)[1]; # split image into HSV channels
    hsv_blur = cv2.blur(hsv_active, (10, 10))  # blur image
 
if __name__ == "__main__":
    main()
