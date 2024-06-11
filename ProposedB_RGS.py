import cv2
import numpy as np
from Code import config as cfg
from Segmentation.BD_COAopt import BD_COA


class BRGS:
    def get8n(self,x, y, shape):
        out = []
        if y-1 > 0 and x-1 > 0:
            out.append( (y-1, x-1) )
        if y-1 > 0 :
            out.append( (y-1, x))
        if y-1 > 0 and x+1 < shape[1]:
            out.append( (y-1, x+1))
        if x-1 > 0:
            out.append( (y, x-1))
        if x+1 < shape[1]:
            out.append( (y, x+1))
        if y+1 < shape[0] and x-1 > 0:
            out.append( ( y+1, x-1))
        if y+1 < shape[0] :
            out.append( (y+1, x))
        if y+1 < shape[0] and x+1 < shape[1]:
           out.append( (y+1, x+1))
        return out

    def region_growing(self,img, seed):
        list = []
        outimg = np.zeros_like(img)

        list.append((seed[0], seed[1]))
        while(len(list)):
            pix = list[0]
            outimg[pix[0], pix[1]] = 255
            for coord in BRGS.get8n(pix[0], pix[1], img.shape):
                if img[coord[0], coord[1]].any() > 0:
                    outimg[coord[0], coord[1]] = 255
                    list.append((coord[0], coord[1]))

            list.pop(0)
        return outimg
    def seg(self,img):
        new_size = (512, 512)
        resize_img = cv2.resize(img, new_size)
        cv2.rectangle(resize_img, (cfg.thres1, cfg.thres2), (cfg.thres3, cfg.thres3),(0, 0, 255), 2)
        cv2.rectangle(resize_img, (cfg.thres5, cfg.thres6), (cfg.thres6, cfg.thres8),(255, 0, 0), 2)
        cv2.rectangle(resize_img, (cfg.thres1, cfg.thres9), (cfg.thres3, cfg.thres8),(0, 255, 0), 2)
        out_img = cv2.rectangle(resize_img, (cfg.thres8, cfg.thres2), (cfg.thres6, cfg.thres3),(155, 155, 0), 2)
        return out_img
    def on_mouse(self,num_coati = 50,max_iter = 100):
        seed = []
        dim = 3
        opt = BD_COA
        fitness = opt.fitness_function
        for i in range(dim - 1):
            print("0, ", end="")
            print("0)")
        best_position = opt.Coati_opt(fitness, max_iter, num_coati, dim, -10.0, 10.0)
        print("\nBest solution found:")
        print(["%.6f" % best_position[k] for k in range(dim)])
        err = fitness(best_position)
        print("fitness of best solution = %.6f" % err)

# image = cv2.imread('..//Check//convexhull.png')
# ret, img = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)
# output = BRGS()
# out_img = output.seg(image)
# cv2.namedWindow('Input')
# cv2.setMouseCallback('Input', on_mouse, 0, )
# cv2.imshow('Input', img)
# cv2.waitKey()
# seed = clicks[0]
# cv2.imshow('Region Growing', region_growing(img, seed))
# cv2.imshow("seg",out_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
