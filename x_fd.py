import cv2
import numpy

# TODO: 
#   improve blur -> use median instead of kernal

from time import strftime

# blur/no blur doesn't result in much difference in terms of image result
# correction: blur is bad. Avoid blur like the plague for MSER

_MYPARAMS = {
    'IMAGE' : "im0216.jpg",
    'HAS_BLUR' : 1,
    'BKS' : 20 # Blur Kernal size
}


def main():
    PRINT_LOG_OUT = []
    PRINT_LOG_OUT.append("\n" + strftime("%Y-%m-%d %H:%M:%S"))
    # print parameters
    PRINT_LOG_OUT += [str(k) + ": "  + str(_MYPARAMS[k]) for k in _MYPARAMS.keys()]

    # import image from file
    imgin = cv2.imread(_MYPARAMS['IMAGE'], cv2.IMREAD_COLOR)
    hsv_imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)

    hsv_chans = cv2.split(hsv_imgin); # split image into HSV channels

    # Get both blurred and not blurred files for image processing
    if (_MYPARAMS['HAS_BLUR']):
        hsv_chans =  [cv2.blur(hsvim, (_MYPARAMS['BKS'], _MYPARAMS['BKS'])) for hsvim in hsv_chans]
    
    # may use other feature detector for testing
    # FD_TYPE = "MSER"
    # PRINT_LOG_OUT.append("FD Type: " + FD_TYPE)
    # my_fd = cv2.MSER_create() 

    FD_TYPE = "SimpleBlob"
    PRINT_LOG_OUT.append("FD Type: " + FD_TYPE)
    my_fd = cv2.SimpleBlobDetector_create() 
    
    kpts = [] # (k)ey(p)oin(t) out
    dkpsout = [] # (d)isplay (k)ey(p)oint (out)put
    for i, im in enumerate(hsv_chans):
        local_kpt = my_fd.detect(im)
        kpts.append([local_kpt])
        if (local_kpt):
            # don't know how the third param works yet  -->
            local_dpksout = cv2.drawKeypoints(im, local_kpt, im) 
            dkpsout.append([local_dpksout]) # append to master list
            cv2.imwrite('dkpsout' + str(i) + '.jpg', local_dpksout)

            # print out num of keypoints and other info 
            PRINT_LOG_OUT.append('Channel: ' + str(i) + ' #kpts: ' + str(len(local_kpt)))

    # output result info to log file
    with open('results.log', 'a') as f:
        for line in PRINT_LOG_OUT:
            f.write(line + '\n')

if __name__ == "__main__":
    main()
