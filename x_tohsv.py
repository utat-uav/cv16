import cv2
import numpy

# Using saturation channel
ACTIVE_CHANNEL = 1

imgin = cv2.imread("im0216.jpg", cv2.IMREAD_COLOR)
hsv_imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)

splitchan = cv2.split(hsv_imgin)

cv2.imshow("imgin", imgin)

for i,im in enumerate(splitchan):
    cv2.imshow("sp_" + str(i), im)
    #cv2.imwrite('outim.jpg', hsv_imgin)

cv2.waitKey(0)
cv2.destroyAllWindows()
