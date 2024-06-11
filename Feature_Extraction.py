import math
import mahotas
import numpy
from PIL import Image, ImageStat, ImageFilter
import numpy as np
from numpy import asarray
from skimage.feature import hog
from statistics import mean
from scipy import ndimage
import csv

Features = []
def features(img2):
    ''' Gradient features'''
    fd, hog_image = hog(img2, orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        visualize=True,
                        multichannel=True)
    hog_val = numpy.nonzero(hog_image)
    sum = 0
    for sub in hog_val:
        for i in sub:
            sum = sum + i
    hog_feature = sum / len(hog_val)
    Features.append(hog_feature)

    ''' Spectral Flatness measure'''
    stat = ImageStat.Stat(img2)
    arith_mean = stat.mean
    SPF = mean(arith_mean)
    Features.append(SPF)

    ''' profile based features'''

    ''' Rib-Cross'''
    image = img2.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)
    rib_cross = asarray(image)
    rib_cross = numpy.nonzero(rib_cross)
    sum = 0
    for sub in rib_cross:
        for i in sub:
            sum = sum + i
    rib_cross = sum / len(rib_cross)
    Features.append(rib_cross)

    '''Peak-ratio'''
    maximum = rib_cross.max()
    minimum = rib_cross.min()
    peak_ratio = (maximum + minimum) / 2
    Features.append(peak_ratio)

    ''' Slope Ratios'''
    image = np.array(img2)
    img = numpy.ndarray.flatten(image)
    var = np.poly1d(img)
    expr_diff = np.gradient(var)
    slope_ratio = numpy.nonzero(expr_diff)
    sum = 0
    for sub in slope_ratio:
        for i in sub:
            sum = sum + i
    slope_ratio = sum / len(slope_ratio)
    Features.append(slope_ratio)

    ''' Slope smooth'''
    slope_smooth = np.gradient(expr_diff)
    slope_smooth = numpy.nonzero(slope_smooth)
    sum = 0
    for sub in slope_smooth:
        for i in sub:
            sum = sum + i
    slope_smooth = sum / len(slope_smooth)
    Features.append(slope_smooth)

    ''' On-Rib feature'''
    im = numpy.nonzero(image)
    value = mahotas.features.eccentricity(im)
    rnds = mahotas.features.roundness(im)
    Features.append(value)
    Features.append(rnds)

    ''' Edge feaures'''
    # Get x-gradient in "sx"
    sx = ndimage.sobel(img2, axis=0, mode='constant')
    # Get y-gradient in "sy"
    sy = ndimage.sobel(img2, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel = np.hypot(sx, sy)
    edge = numpy.nonzero(sobel)
    sum = 0
    for sub in edge:
        for i in sub:
            sum = sum + i
    edge = sum / len(edge)
    Features.append(edge)

    ''' on-vessel'''
    length = np.sum(image == 255)
    h, w, c = image.shape
    vessel1 = length / (h * w)
    Features.append(vessel1)

    red = np.array([255, 0, 0], dtype=np.uint8)
    reds = np.where(np.all((image == red), axis=-1))
    blue = np.array([0, 0, 255], dtype=np.uint8)
    blues = np.where(np.all((image == blue), axis=-1))
    distance1 = []
    for i in range(len(reds)):
        for j in range(len(blues)):
            dx2 = (reds[i][j] - reds[i][j]) ** 2  # (200-10)^2
            dy2 = (blues[i][j] - reds[i][j]) ** 2  # (300-20)^2
            distance = math.sqrt(dx2 + dy2)
            distance1.append(distance)
    distance_val = min(distance1)
    distance_val = 1 / distance_val
    vessel2 = 1.0 / distance_val
    Features.append(vessel2)
    # fields = ["Gradient", "Spectral Flatness","Rib-cross", "Peak-ratio", "Slope-ratio","Slope-smooth", "On-Rib Rands","On-Rib value", "Edge", "Vessel1", "vessel2"]
    # filename = "Features.csv"
    # # writing to csv file
    # with open(filename, 'w') as csvfile:
    #     # creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #
    #     # writing the fields
    #     csvwriter.writerow(fields)
    #
    #     # writing the data rows
    #     csvwriter.writerow(Features)

# img2 = Image.open("..//Image_output//Segmented_image.png")
# features(img2)