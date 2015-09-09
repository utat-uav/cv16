import cv2
import numpy
import os
import shutil
import copy

# TODO: 
#   Add false positive filter

from time import strftime

# blur/no blur doesn't result in much difference in terms of image result
# correction: blur is bad. Avoid blur like the plague for MSER

_MYPARAMS = {
    'ACTIVE_CHANNEL' : 2,
    'IMAGE' : "im0211.jpg",
    'HAS_BLUR' : 1,
    'BKS' : 7, # Blur Kernal size
    'SIZE_OF_ROI' : 120, # Size of target to crop
    'MIN_POINTS_IN_CLUSTER' : 1
}

def clamp(num, mymin, mymax):
    return min(mymax, max(num, mymin))


def not_too_close(new_pt, ptlist):
    # minimum taxicab distance must be greater than size of roi
    # detect if two points are too close
    sroi = _MYPARAMS['SIZE_OF_ROI']
    for pt in ptlist:
        if (abs(new_pt[0] - pt[0]) < sroi or abs(new_pt[1] - pt[1]) < sroi):
            return 0
    # if pointlist is empty
    return 1

def distanceBetween(one, two):
    return ((one[0] - two[0]) ** 2 + (one[1] - two[1]) ** 2) ** 0.5

def cluster(kpts):
    clusters = []
    clusterIndex = 0
    checkedPoints = [-1] * len(kpts) # Stores the index of the cluster that it belongs to

    for i, kpt in enumerate(kpts):
        if checkedPoints[i] == -1: # If the point hasn't been checked yet
            point = (int(kpt.pt[1]), int(kpt.pt[0]))

            checkedPoints[i] = clusterIndex
            # Create a new cluster for it
            clusters.append([point])

            # Check to see if this point belongs to any other clusters
            pointSorted = False
            for j, pointClusterIndex in enumerate(checkedPoints):
                if pointClusterIndex > -1 and j != i: # If this point has been put into a cluster and isn't itself
                    pointCheck = (int(kpts[j].pt[1]), int(kpts[j].pt[0]))
                    distance = distanceBetween(pointCheck, point)
                    if distance <= _MYPARAMS['SIZE_OF_ROI']:
                        # Add to this pointClusterIndex
                        clusters[pointClusterIndex].append(pointCheck)
                        # Remove the current cluster
                        clusters.pop()
                        clusterIndex = clusterIndex - 1
                        pointSorted = True
                        break

            # If this point is destined to be a whole new group
            if not pointSorted:
                for j, kptCheck in enumerate(kpts):
                    pointCheck = (int(kptCheck.pt[1]), int(kptCheck.pt[0]))
                    distance = distanceBetween(pointCheck, point)
                    if checkedPoints[j] == -1 and distance <= _MYPARAMS['SIZE_OF_ROI']: # If not checked and within suitable distance
                        checkedPoints[j] = clusterIndex
                        clusters[clusterIndex].append(pointCheck)

            clusterIndex = clusterIndex + 1  
    
    filteredClusters = []
    # find average amount of clusters
    lengths = []
    for cluster in clusters:
        lengths.append(len(cluster))
    avgLen = numpy.mean(lengths)
    stdLen = numpy.std(lengths)

    for cluster in clusters:
        if len(cluster) > _MYPARAMS['MIN_POINTS_IN_CLUSTER'] and len(cluster) > (avgLen): # plus stdLen?
            filteredClusters.append(cluster)

    if len(filteredClusters) >= 1:
        return filteredClusters # Indicates high confidence in results
    return clusters # Indicates low confidence in results

def averageClusters(clusters):
    averagedClusters = []
    for i, cluster in enumerate(clusters):
        avgX = 0
        avgY = 0
        # sum points
        for point in cluster:
            avgX = avgX + point[0]
            avgY = avgY + point[1]
        # average points
        avgX = avgX / len(cluster)
        avgY = avgY / len(cluster)
        # add to list of clusters
        averagedClusters.append((int(avgX), int(avgY)))

    return averagedClusters

def main():
    PRINT_LOG_OUT = []
    PRINT_LOG_OUT.append("\n" + strftime("%Y-%m-%d %H:%M:%S"))
    # print parameters
    PRINT_LOG_OUT += [str(k) + ": "  + str(_MYPARAMS[k]) for k in _MYPARAMS.keys()]
    
    # output folder
    fileList = os.listdir("Output")
    for fileName in fileList:
        os.remove("Output/"+fileName)
    
    # import image from file
    imgin = cv2.imread(_MYPARAMS['IMAGE'], cv2.IMREAD_COLOR)
    hsv_imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)

    # Detect image size [rows, columns]
    _IMBND = (hsv_imgin.shape[0], hsv_imgin.shape[1])

    hsv_chans = cv2.split(hsv_imgin); # split image into HSV channels

    # Get both blurred and not blurred files for image processing
    if (_MYPARAMS['HAS_BLUR']):
        hsv_chans =  [cv2.blur(hsvim, (_MYPARAMS['BKS'], _MYPARAMS['BKS'])) for hsvim in hsv_chans]
    
    # may use other feature detector for testing
    FD_TYPE = "MSER"
    PRINT_LOG_OUT.append("FD Type: " + FD_TYPE)
    my_fd = cv2.MSER_create() 

    # FD_TYPE = "SimpleBlob"
    # PRINT_LOG_OUT.append("FD Type: " + FD_TYPE)
    # my_fd = cv2.SimpleBlobDetector_create() 
    
    kpts = [] # (k)ey(p)oin(t) out
    dkpsout = [] # (d)isplay (k)ey(p)oint (out)put
    for i, im in enumerate(hsv_chans):
        local_kpt = my_fd.detect(im) # local keypoints

        imgClusteredRegions = copy.copy(imgin)
        cv2.drawKeypoints(imgClusteredRegions, local_kpt, imgClusteredRegions)

        kpts.append(local_kpt)
        if (local_kpt):
            # don't know how the third param works yet  -->
            local_dpksout = cv2.drawKeypoints(im, local_kpt, im) 
            dkpsout.append([local_dpksout]) # append to master list
            cv2.imwrite(os.path.join("Output", 'dkpsout' + str(i) + '.jpg'), local_dpksout)

            # print out num of keypoints and other info 
            PRINT_LOG_OUT.append('Channel: ' + str(i) + ' #kpts: ' + str(len(local_kpt)))

    # Crop out ROIs for active_channel
    # TODO: Add to log file
    ptlist = []
    clusters = cluster(kpts[_MYPARAMS['ACTIVE_CHANNEL']])
    clusters = averageClusters(clusters)
    hs = _MYPARAMS['SIZE_OF_ROI'] / 2
    for i, mypoint in enumerate(clusters):
        row_crop = (clamp(mypoint[0]-hs, 0, _IMBND[0]), clamp(mypoint[0]+hs, 0, _IMBND[0]))
        col_crop = (clamp(mypoint[1]-hs, 0, _IMBND[1]), clamp(mypoint[1]+hs, 0, _IMBND[1]))
        new_crop = imgin[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]
        cv2.imwrite(os.path.join("Output", 'chan' + str(_MYPARAMS['ACTIVE_CHANNEL']) + '_roi' + str(i) + '.jpg'), new_crop)
        # draw rectangle
        cv2.rectangle(imgClusteredRegions, (mypoint[1]-hs, mypoint[0]-hs), (mypoint[1]+hs,mypoint[0]+hs), (255, 255, 255), 1)

    cv2.imwrite(os.path.join("Output", 'croppedRegions.jpg'), imgClusteredRegions)

    ''' OLD METHOD
    for i, kpt in enumerate(kpts[_MYPARAMS['ACTIVE_CHANNEL']]):
        # MSER detects features as a fraction of a coordinate, apparently
        # Also, cv2 keypoints are a pain to work with, so I'm turning it into a regular old tuple
        # Should be (row, column) as usual
        mypoint = (int(kpt.pt[1]), int(kpt.pt[0]))

        if not_too_close(mypoint, ptlist) == 1:
            hs = _MYPARAMS['SIZE_OF_ROI'] / 2 # (h)alf (s)ize
            # Schenanigans for cropping and making sure that crop doesn't exceed bounds
            row_crop = (clamp(mypoint[0]-hs, 0, _IMBND[0]), clamp(mypoint[0]+hs, 0, _IMBND[0]))
            col_crop = (clamp(mypoint[1]-hs, 0, _IMBND[1]), clamp(mypoint[1]+hs, 0, _IMBND[1]))
            new_crop = imgin[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]

            cv2.imwrite(os.path.join("Output", 'chan' + str(_MYPARAMS['ACTIVE_CHANNEL']) + '_roi' + str(i) + '.jpg'), new_crop)
            ptlist.append(mypoint) 
    END OF OLD METHOD ''' 

    # print result info to log file
    with open('results.log', 'a') as f:
        for line in PRINT_LOG_OUT:
            f.write(line + '\n')

if __name__ == "__main__":
    main()
