import cv2
import matplotlib.pyplot as plt
from PIL import Image as im
import skimage.filters as flt
import numpy as np
import warnings
import scipy.ndimage as flt


def eanisodifff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0.5, option=1,ploton=False):

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()


    for ii in np.arange(1, niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if 0 < sigma:
            deltaSf = flt.gaussian_filter(deltaS, sigma);
            deltaEf = flt.gaussian_filter(deltaE, sigma);
        else:
            deltaSf = deltaS;
            deltaEf = deltaE;

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaEf / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaSf / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaEf / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout

# img=cv2.imread('..//Dataset//Data//train//normal//n7.png')
# img = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
# img=img.astype(float)
# # # img=img[300:600,300:600]
# m=np.mean(img)
# s=np.std(img)
# nimg=(img-m)/s
# fimg=anisodifff(nimg,100,80,0.0075,(1,1),2.5,1)
# result = im.fromarray((fimg*255).astype(np.uint8))
# result.show('noiseremoved.png')


