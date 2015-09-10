import cv2
import numpy
import os
import shutil
import copy
import random

# TODO: 
#   Add false positive filter

from time import strftime

# blur/no blur doesn't result in much difference in terms of image result
# correction: blur is bad. Avoid blur like the plague for MSER

_MYPARAMS = {
    'ACTIVE_CHANNEL' : -1,
    'IMAGE' : "im0211.jpg",
    'HAS_BLUR' : 1,
    'BKS' : 6, # Blur Kernal size
    'SIZE_OF_ROI' : 120, # Size of target to crop
    'MIN_POINTS_IN_CLUSTER' : 1
}

'''Used to make sure that a point is within a certain ROI'''
def clamp(num, mymin, mymax):
    return min(mymax, max(num, mymin))

'''Colorful drawings'''
def drawClusters(imgClusteredRegions, clusters, clusterLocations):
    for i, cluster in enumerate(clusters):
        #Get point
        clusterLocation = clusterLocations[i]
        hs = _MYPARAMS['SIZE_OF_ROI']/2
        # Draw rectangle
        cv2.rectangle(imgClusteredRegions, (clusterLocation[1]-hs, clusterLocation[0]-hs), (clusterLocation[1]+hs,clusterLocation[0]+hs), (255, 255, 255), 1)
        # Draw number of dots
        cv2.putText(imgClusteredRegions, str(len(cluster)) + ' points', (clusterLocation[1]-hs, clusterLocation[0]-hs), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        # Draw dots
        randClusterColor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for clusterPoint in cluster:
            cv2.circle(imgClusteredRegions, (int(clusterPoint[1]), int(clusterPoint[0])), 4, randClusterColor, 2)
    return imgClusteredRegions

'''Pythag it'''
def distanceBetween(one, two):
    return ((one[0] - two[0]) ** 2 + (one[1] - two[1]) ** 2) ** 0.5

'''RECURSIVE METHOD'''
# Global variables
checkedPoints = []
maxDist = _MYPARAMS['SIZE_OF_ROI']/2
maxClusterIndex = -1

def copyKpts(kpts):
    global checkedPoints
    for i, kpt in enumerate(kpts):
        point = (int(kpt.pt[1]), int(kpt.pt[0]))
        checkedPoints.append([point[0], point[1], -1]) # X, Y, ClusterIndex

def getClosestNeighbours(j, clustered):
    global checkedPoints
    minDist = -1
    closestNeighbour = -1
    closestNeighbours = []
    for i, pointCheck in enumerate(checkedPoints):
        if i != j:
            distance = distanceBetween((checkedPoints[j][0], checkedPoints[j][1]), (pointCheck[0], pointCheck[1]))
            mode = True
            if clustered:
                mode = pointCheck[2] > -1
            else:
                mode = pointCheck[2] == -1
            if mode and distance <= maxDist: # If not itself and not clustered and within a fair distance
                closestNeighbours.append(i) # stores pointIndices
                if minDist == -1:
                    minDist = distance
                if distance <= minDist:
                    minDist = distance
                    closestNeighbour = i
    return closestNeighbour, closestNeighbours

def checkPoint(i):
    global checkedPoints
    global maxClusterIndex

    # Set the cluster index
    checkedPoints[i][2] = maxClusterIndex

    # Get neighbours
    closestUnclusteredNeighbour, closestUnclusteredNeighbours = getClosestNeighbours(i, False)

    for unclusteredNeighbour in closestUnclusteredNeighbours:
        # Recursive call for neighbours
        neighbourClustered = checkPoint(unclusteredNeighbour)

def getClusters(kpts):
    global checkedPoints
    global maxClusterIndex
    copyKpts(kpts)
    for i, point in enumerate(checkedPoints):
        if point[2] == -1:
            maxClusterIndex += 1
            checkPoint(i)

    clusters = []
    for i in range(maxClusterIndex):
        clusters.append([])
        for j, point in enumerate(checkedPoints):
            if point[2] == i:
                clusters[i].append((point[0], point[1]))

    # Remove high STD values
    clusterSTDs = stdClusters(clusters)
    stdFilteredClusters = []
    for i, cluster in enumerate(clusters):
        if clusterSTDs[i][0] < _MYPARAMS['SIZE_OF_ROI']/2 and clusterSTDs[i][1] < _MYPARAMS['SIZE_OF_ROI']/2: # plus stdLen?
            stdFilteredClusters.append(cluster)
    clusters = stdFilteredClusters

    filteredClusters = []
    # find average amount of clusters
    lengths = []
    for cluster in clusters:
        lengths.append(len(cluster))
    avgLen = numpy.mean(lengths)
    stdLen = numpy.std(lengths)

    for i, cluster in enumerate(clusters):
        if len(cluster) > _MYPARAMS['MIN_POINTS_IN_CLUSTER'] and len(cluster) > (avgLen): # plus stdLen?
            filteredClusters.append(cluster)

    if len(filteredClusters) >= 1:
        return len(clusters) - len(filteredClusters), filteredClusters # Indicates high confidence in results
    return 0, clusters # Indicates low confidence in results

'''END RECURSIVE METHOD'''

'''Average the location of all clusters'''
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

'''Get standard deviation of clusters'''
def stdClusters(clusters):
    clusterStds = []
    for i, cluster in enumerate(clusters):
        xVals = []
        yVals = []
        for point in cluster:
            xVals.append(point[0])
            yVals.append(point[1])
        xSTD = numpy.std(xVals)
        ySTD = numpy.std(yVals)
        clusterStds.append([xSTD, ySTD])

    return clusterStds

def main():
    PRINT_LOG_OUT = []
    PRINT_LOG_OUT.append("[Date]")
    PRINT_LOG_OUT.append("Date = " + strftime("%Y-%m-%d %H:%M:%S"))
    PRINT_LOG_OUT.append("\n[Analysis Parameters]")
    # print parameters
    PRINT_LOG_OUT += [str(k) + " = "  + str(_MYPARAMS[k]) for k in _MYPARAMS.keys()]
    
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
    PRINT_LOG_OUT.append("FD Type = " + FD_TYPE)
    my_fd = cv2.MSER_create() 

    # FD_TYPE = "SimpleBlob"
    # PRINT_LOG_OUT.append("FD Type: " + FD_TYPE)
    # my_fd = cv2.SimpleBlobDetector_create() 
    
    imgClusteredRegions = copy.copy(imgin)

    PRINT_LOG_OUT.append("\n[Channel Keypoints]")

    kpts = [] # (k)ey(p)oin(t) out
    dkpsout = [] # (d)isplay (k)ey(p)oint (out)put
    for i, im in enumerate(hsv_chans):
        local_kpt = my_fd.detect(im) # local keypoints

        if _MYPARAMS['ACTIVE_CHANNEL'] == -1:
            for point in local_kpt:
                kpts.append(point)
        else:
            kpts.append(local_kpt)
        if (local_kpt):
            # don't know how the third param works yet  -->
            local_dpksout = cv2.drawKeypoints(im, local_kpt, im) 
            dkpsout.append([local_dpksout]) # append to master list
            cv2.imwrite(os.path.join("Output", 'dkpsout' + str(i) + '.jpg'), local_dpksout)

            # print out num of keypoints and other info 
            PRINT_LOG_OUT.append('Channel ' + str(i) + ' = ' + str(len(local_kpt)))

    # Crop out ROIs for active_channel
    # TODO: Add to log file
    if _MYPARAMS['ACTIVE_CHANNEL'] == -1:
        numRejected, clusters = getClusters(kpts)
    else:
        numRejected, clusters = getClusters(kpts[_MYPARAMS['ACTIVE_CHANNEL']])
    averagedClusters = averageClusters(clusters)
    clusterSTDs = stdClusters(clusters)
    hs = _MYPARAMS['SIZE_OF_ROI'] / 2
    for i, mypoint in enumerate(averagedClusters):
        row_crop = (clamp(mypoint[0]-hs, 0, _IMBND[0]), clamp(mypoint[0]+hs, 0, _IMBND[0]))
        col_crop = (clamp(mypoint[1]-hs, 0, _IMBND[1]), clamp(mypoint[1]+hs, 0, _IMBND[1]))
        new_crop = imgin[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]
        cv2.imwrite(os.path.join("Output", 'chan' + str(_MYPARAMS['ACTIVE_CHANNEL']) + '_roi' + str(i) + '.jpg'), new_crop)

    # Log clustering info
    confidence = ""
    if numRejected == 0:
        confidence = "Low"
    else:
        confidence = "High"
    PRINT_LOG_OUT.append("\n[Clustering Info]")
    PRINT_LOG_OUT.append("Rejected Clusters = " + str(numRejected))
    PRINT_LOG_OUT.append("Confidence = " + confidence)

    # Write log for the clusters
    for i, cluster in enumerate(clusters):
        PRINT_LOG_OUT.append("\n[Cluster " + str(i+1) + "]")
        PRINT_LOG_OUT.append("NumPoints = " + str(len(cluster)))
        PRINT_LOG_OUT.append("X = " + str(averagedClusters[i][1]))
        PRINT_LOG_OUT.append("Y = " + str(averagedClusters[i][0]))
        PRINT_LOG_OUT.append("X STD = " + str(clusterSTDs[i][1]))
        PRINT_LOG_OUT.append("Y STD = " + str(clusterSTDs[i][0]))

    # Output cluster locations
    imgClusteredRegions = drawClusters(imgClusteredRegions, clusters, averagedClusters)
    cv2.imwrite(os.path.join("Output", 'croppedRegions.jpg'), imgClusteredRegions)

    # print result info to log file
    with open(os.path.join("Output", 'results.ini'), 'a') as f:
        for line in PRINT_LOG_OUT:
            f.write(line + '\n')

if __name__ == "__main__":
    main()
