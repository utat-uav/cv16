import cv2
import numpy
import os
import shutil
import copy
import random
from time import strftime

# TODO: 
#   Improve false positive filter
#   Possibly using standard deviation of hue variance (compare the std of cropped image to master image or only the std within the cropped image?)
#   Test if making the BKS (blur amount) proportional to the standard deviation in contrast works

_MYPARAMS = {
    'ACTIVE_CHANNEL' : [1,2],
    'IMAGE' : "IMG_0520.jpg",
    'HAS_BLUR' : 1,
    'BKS' : 6, # Blur Kernal size
    'SIZE_OF_ROI' : 300, # Cluster size
    'MIN_POINTS_IN_CLUSTER' : 5,
    'USE_TREE_FILTER' : 1, # Filters out crops that are "tree colored"
    'MAX_AREA' : 38000,
    'MIN_AREA' : 3500
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

def drawCroppedRegions(imgCroppedRegions, locations, sizes):
    for i, location in enumerate(locations):
        hs = int((sizes[i][0] if sizes[i][0] > sizes[i][1] else sizes[i][1])/2)
        cv2.rectangle(imgCroppedRegions, (int(location[1])-hs, int(location[0])-hs), (int(location[1])+hs, int(location[0])+hs), (255, 255, 255), 1)
    return imgCroppedRegions

'''Pythag it'''
def distanceBetween(one, two):
    return ((one[0] - two[0]) ** 2 + (one[1] - two[1]) ** 2) ** 0.5

'''Converts hulls to useful data'''
def hulls2Points(hulls):
    points = []
    sizes = []

    for hull in hulls:
        maxX = 0
        minX = 100000
        maxY = 0
        minY = 100000
        x = []
        y = []
        for point in hull:
            if point[0][0] > maxX:
                maxX = point[0][0]
            if point[0][0] < minX:
                minX = point[0][0]
            if point[0][1] > maxY:
                maxY = point[0][1]
            if point[0][1] < minY:
                minY = point[0][1]
            x.append(point[0][0])
            y.append(point[0][1])
        points.append([numpy.mean(x), numpy.mean(y)])
        sizes.append([maxX-minX, maxY-minY])

    return points, sizes

def largestSize(clusters, sizes):

    #detailedCoordinates = []
    largestSizes = []
    for i, cluster in enumerate(clusters):
        #detailedCoordinates.append([])
        minX = 1000000
        minY = 1000000
        maxX = 0
        maxY = 0
        for j, point in enumerate(cluster):
            if point[0]-(sizes[i][j][0])/2 < minX:
                minX = point[0]-(sizes[i][j][0])/2
            if point[0]+(sizes[i][j][0])/2 > maxX:
                maxX = point[0]+(sizes[i][j][0])/2
            if point[1]-(sizes[i][j][1])/2 < minY:
                minY = point[1]-(sizes[i][j][1])/2
            if point[1]+(sizes[i][j][1])/2 > maxY:
                maxY = point[1]+(sizes[i][j][1])/2
            #detailedCoordinates[i].append([point[0]-(sizes[i][j][0])/2, point[0]+(sizes[i][j][0])/2, point[1]-(sizes[i][j][1])/2, point[1]+(sizes[i][j][1])/2]) #minX, maxX, minY, maxY
        largestSizes.append([maxX - minX, maxY - minY])

    #for i, size in enumerate(sizes):
    #    maxX = 0
    #    maxY = 0
    #    for point in size:
    #        if point[0] > maxX:
    #            maxX = point[0]
    #        if point[1] > maxY:
    #            maxY = point[1]
    #    largestSizes.append([maxX, maxY])
    return largestSizes

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''DBSCAN'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Global variables
checkedPoints = []
maxDist = _MYPARAMS['SIZE_OF_ROI']/2
maxClusterIndex = -1

'''Converts raw keypoints from openCV to a regular array'''
def copyKpts(kpts):
    global checkedPoints
    for i, kpt in enumerate(kpts):
        point = (int(kpt.pt[1]), int(kpt.pt[0]))
        checkedPoints.append([point[0], point[1], -1]) # X, Y, ClusterIndex

def initPoints(kpts):
    global checkedPoints
    for i, kpt in enumerate(kpts):
        checkedPoints.append([kpt[1], kpt[0], -1]) # X, Y, ClusterIndex

'''Region query'''
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

'''Expanding for DBSCAN algorithm'''
def checkPoint(i):
    global checkedPoints
    global maxClusterIndex

    # Set the cluster index
    checkedPoints[i][2] = maxClusterIndex

    # Get neighbours / do region query for unclustered neighbours
    closestUnclusteredNeighbour, closestUnclusteredNeighbours = getClosestNeighbours(i, False)

    k = 0
    while k < len(closestUnclusteredNeighbours):
        unclusteredNeighbour = closestUnclusteredNeighbours[k]
        k += 1

        # Recursive call for neighbours
        #neighbourClustered = checkPoint(unclusteredNeighbour)

        # Non-recursive method
        checkedPoints[unclusteredNeighbour][2] = maxClusterIndex
        newClosestUnclusteredNeighbour, newClosestUnclusteredNeighbours = getClosestNeighbours(unclusteredNeighbour, False)
        # Combine without repeats
        closestUnclusteredNeighbours = closestUnclusteredNeighbours + list(set(newClosestUnclusteredNeighbours) - set(closestUnclusteredNeighbours))

'''Runs DBSCAN and false positive filter'''
def getClusters(kpts, kptsSizes):
    global checkedPoints
    global maxClusterIndex
    #copyKpts(kpts)
    initPoints(kpts)
    for i, point in enumerate(checkedPoints):
        if point[2] == -1:
            maxClusterIndex += 1
            checkPoint(i)

    clusters = []
    clusterSizes = []
    for i in range(maxClusterIndex + 1):
        clusters.append([])
        clusterSizes.append([])
        for j, point in enumerate(checkedPoints):
            if point[2] == i:
                clusters[i].append((point[0], point[1]))
                clusterSizes[i].append((kptsSizes[j][0], kptsSizes[j][1]))

    filteredClusters = []
    filteredClusterSizes = []
    # find average amount of clusters
    lengths = []
    for cluster in clusters:
        lengths.append(len(cluster))
    avgLen = numpy.mean(lengths)
    stdLen = numpy.std(lengths)

    for i, cluster in enumerate(clusters):
        #if len(cluster) >= _MYPARAMS['MIN_POINTS_IN_CLUSTER'] and len(cluster) >= (avgLen + stdLen): # plus stdLen?
        #if len(cluster) <= avgLen + stdLen:
        filteredClusters.append(cluster)
        filteredClusterSizes.append(clusterSizes[i])

    return filteredClusters, filteredClusterSizes # Indicates high confidence in results

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''END OF DBSCAN'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''Filter for trees by color'''
def filterTrees(imgin, clusters, clusterSizes):
    filteredClusters = []
    filteredClusterSizes = []

    _IMBND = (imgin.shape[0], imgin.shape[1])
    avgColor = [0, 0, 0]
    for i, mypoint in enumerate(clusters):
        cropSize = clusterSizes[i][1]/2 if clusterSizes[i][1]/2 > clusterSizes[i][0]/2 else clusterSizes[i][0]/2
        row_crop = (clamp(mypoint[0]-cropSize, 0, _IMBND[0]), clamp(mypoint[0]+cropSize, 0, _IMBND[0]))
        col_crop = (clamp(mypoint[1]-cropSize, 0, _IMBND[1]), clamp(mypoint[1]+cropSize, 0, _IMBND[1]))
        new_crop = imgin[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]

        # Create solid image and threshold
        new_crop = cv2.cvtColor(new_crop, cv2.COLOR_BGR2HSV)
        averagePixel = cv2.mean(new_crop)
        avgColor = [avgColor[0]+averagePixel[0], avgColor[1]+averagePixel[1], avgColor[2]+averagePixel[2]]
        if not (averagePixel[0] > 17 and averagePixel[0] < 58 and averagePixel[1] > 10 and averagePixel[1] < 105 and averagePixel[2] > 0 and averagePixel[2] < 103):
            # Add to return list
            filteredClusters.append(mypoint)
            filteredClusterSizes.append(clusterSizes[i])
    
    #34.5, 58.5, 48.2 in HSV approximately make tree-green
    return filteredClusters, filteredClusterSizes

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
    PRINT_LOG_OUT.append("Date Analyzed = " + strftime("%Y-%m-%d %H:%M:%S"))
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
    # delta, maxArea, minArea, maxVariation, minDiversity, maxEvolution, areaThreshold, minMargin, edgeBlurSize
    # Decreasing maxVariation increases how sharp edges need to be
    my_fd = cv2.MSER_create(5, _MYPARAMS['MIN_AREA'] / 2738 * _IMBND[0], _MYPARAMS['MAX_AREA'] / 2738 * _IMBND[0], 0.099, 0.65, 200, 1.01, 0.003, 5) # Default is 5, 60, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5

    # FD_TYPE = "SimpleBlob"
    # PRINT_LOG_OUT.append("FD Type: " + FD_TYPE)
    # my_fd = cv2.SimpleBlobDetector_create() 
    
    imgClusteredRegions = copy.copy(imgin)

    PRINT_LOG_OUT.append("\n[Channel Keypoints]")

    kpts = [] # (k)ey(p)oin(t) out
    kptsSize = []
    dkpsout = [] # (d)isplay (k)ey(p)oint (out)put
    for i, im in enumerate(hsv_chans):
        local_kpt = my_fd.detect(im, None) # local keypoints

        if len([x for x in _MYPARAMS['ACTIVE_CHANNEL'] if x == i]) > 0:
            # Outputs image of regions
            vis = im.copy()
            regions = my_fd.detectRegions(im, None)
            hulls = [cv2.convexHull(s.reshape(-1, 1, 2)) for s in regions]
            hullLocations, hullSizes = hulls2Points(hulls)
            cv2.polylines(vis, hulls, 1, (0, 255, 0))
            cv2.imwrite(os.path.join("Output", 'region visualization' + str(i) + '.jpg'), vis)

            for i, point in enumerate(hullLocations):
                kpts.append(point)
                kptsSize.append(hullSizes[i])

        if (local_kpt):
            # don't know how the third param works yet  -->
            local_dpksout = cv2.drawKeypoints(im, local_kpt, im) 
            dkpsout.append([local_dpksout]) # append to master list
            cv2.imwrite(os.path.join("Output", 'dkpsout' + str(i) + '.jpg'), local_dpksout)

            # print out num of keypoints and other info 
            PRINT_LOG_OUT.append('Channel ' + str(i) + ' = ' + str(len(local_kpt)))

    # Crop out ROIs for active_channel
    clusters, clusterSizes = getClusters(kpts, kptsSize)
    averagedClusters = averageClusters(clusters)
    clusterSizes = largestSize(clusters, clusterSizes)
    # Tree filter
    if _MYPARAMS['USE_TREE_FILTER']:
        averagedClusters, clusterSizes = filterTrees(imgin, averagedClusters, clusterSizes)

    croppedImgNames = []
    for i, mypoint in enumerate(averagedClusters):
        cropSize = clusterSizes[i][1]/2 if clusterSizes[i][1]/2 > clusterSizes[i][0]/2 else clusterSizes[i][0]/2
        row_crop = (clamp(mypoint[0]-cropSize, 0, _IMBND[0]), clamp(mypoint[0]+cropSize, 0, _IMBND[0]))
        col_crop = (clamp(mypoint[1]-cropSize, 0, _IMBND[1]), clamp(mypoint[1]+cropSize, 0, _IMBND[1]))
        new_crop = imgin[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]
        croppedImgNames.append('chan' + str(_MYPARAMS['ACTIVE_CHANNEL']) + '_roi' + str(i) + '.jpg')
        cv2.imwrite(os.path.join("Output", croppedImgNames[i]), new_crop)

    # Log clustering info
    PRINT_LOG_OUT.append("\n[Crop Info]")
    PRINT_LOG_OUT.append("Number of Crops = " + str(len(averagedClusters)))

    # Write log for the clusters
    for i, cluster in enumerate(averagedClusters):
        PRINT_LOG_OUT.append("\n[Crop " + str(i+1) + "]")
        PRINT_LOG_OUT.append("Image Name = " + croppedImgNames[i])
        PRINT_LOG_OUT.append("X = " + str(averagedClusters[i][1]))
        PRINT_LOG_OUT.append("Y = " + str(averagedClusters[i][0]))

    # Output cluster locations
    #imgClusteredRegions = drawClusters(imgClusteredRegions, clusters, averagedClusters)
    imgClusteredRegions = drawCroppedRegions(imgClusteredRegions, averagedClusters, clusterSizes)
    cv2.imwrite(os.path.join("Output", 'croppedRegions.jpg'), imgClusteredRegions)

    # print result info to log file
    with open(os.path.join("Output", 'results.ini'), 'a') as f:
        for line in PRINT_LOG_OUT:
            f.write(line + '\n')

if __name__ == "__main__":
    main()
