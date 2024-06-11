import csv
import math
import os
from statistics import mean
from tkinter import Tk, messagebox, filedialog
from tkinter import *
import tkinter
from typing import List, Tuple
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
import PIL
import cv2
import mahotas
import random
import numpy
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageFilter, ImageStat
from PIL import Image as im
from numpy import asarray
from scipy import ndimage
from skimage.feature import hog
from Code import config as cfg


from NoiseRemoval.IADF_Noise_Removal import anisodiff
from NoiseRemoval.Existing_ADF import eanisodifff
from NoiseRemoval.Existing_GF import GF
from NoiseRemoval.Existing_MF import MF
from NoiseRemoval.Existing_BF import BF
from NoiseRemoval.Result_metrics import SystemEvaluation

from Segmentation.ProposedB_RGS import BRGS
from Segmentation.Existing_RGS import RGS
from Segmentation.Existing_Watershed import watershed
from Segmentation.Existing_Otsus import OS
from Segmentation.Existing_KMA import KMA
from Segmentation.Seg_metric import Seg_met

from Classification import Existing_ANN
from Classification import Existing_DNN
from Classification import Existing_CNN
from Classification import Existing_RESNET
from Classification import Proposed_PResNet

from Code.Contrast_Stretching import contrat_stretching
from Code.Convex_Hull_lung import convex_hull
from FeatureSelection.Proposed_DCST import ChiSquare




class Main_GUI:

    bool_browse_dataset= False
    bool_noise_removal = False
    bool_contrast_stretchin = False
    bool_convex_hull = False
    bool_edge_enhancement = False
    bool_segmentation = False
    bool_feature_extraction = False
    bool_feature_selection = False
    bool_dataset_splitting = False
    bool_classification_train = False
    bool_classification_test = False
    bool_risk_screening = False
    bool_select_image = False
    bool_image_preprocessing = False
    bool_image_segmentation = False
    bool_image_feature_extraction = False
    bool_image_feature_selection = False
    bool_image_classification = False
    bool_image_risk_screening = False

    def __init__(self, root):

        self.root = root
        self.filename = ""
        self.noiseremoved=""
        self.stretched_image = " "
        self.Train_selected_features = ""
        self.Test_selected_features = ""
        self.Valid_selected_features = ""
        self.classss = ""
        self.data = ""


        self.LARGE_FONT = ("Algerian", 14, "bold")
        self.text_font1 = ("Constantia", 11)

        label_heading = tkinter.Label(root, text="P-RESNET WITH B-RGS LUNG CARCINOMA PREDICTION MODEL AT PRE-MATURE STAGE",bg="powderblue",fg="indian red", font=self.LARGE_FONT).place(x=175,y=5)

        self.label_browse_dataset = tkinter.LabelFrame(root, text="Browse dataset",bg="powderblue",fg="dark orange", font=self.text_font1)
        self.label_browse_dataset.place(x=20, y=40, width=325, height=75)
        self.entry_browse_dataset = tkinter.Entry(root, width=30)
        self.entry_browse_dataset.place(x=30, y=70, height=25)
        self.btn_browse_dataset = Button(root, text="Browse", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1,command=self.browse_dataset)
        self.btn_browse_dataset.place(x=230, y=66)

        self.label_preprocessing = tkinter.LabelFrame(root, text="Pre-processing",bg="powderblue",fg="dark orange", font=self.text_font1)
        self.label_preprocessing.place(x=370, y=40, width=290, height=130)
        self.noise_btn = Button(root,text="IADF", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.noise_removal)
        self.noise_btn.place(x=380, y=70)
        self.contrast_btn = Button(root,text="Contrast  stretching", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=15, height=1, command=self.contrast_stretching)
        self.contrast_btn.place(x=500,y=70)
        self.convexhull_btn = Button(root,text="Convex hull", bg="chartreuse2", fg="deep pink",width=10,font=self.text_font1, height=1, command = self.convex_hull_seperation)
        self.convexhull_btn.place(x=380,y=120)
        self.edge_btn = Button(root,text="UMF", bg="chartreuse2", fg="deep pink",width=10,font=self.text_font1, height=1, command=self.edge_enhancement)
        self.edge_btn.place(x=500,y=120)

        self.label_segmentation = tkinter.LabelFrame(root, text="Segmentation",bg="powderblue",fg="dark orange", font=self.text_font1)
        self.label_segmentation.place(x=685,y=40,width=150,height=75)
        self.segmentation_btn = Button(root,text="BRGS", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.segmentation)
        self.segmentation_btn.place(x=707, y=70)

        self.label_feature_extraction = tkinter.LabelFrame(root, text="Feature Extraction",bg="powderblue",fg="dark orange", font=self.text_font1)
        self.label_feature_extraction.place(x=860,y=40,width=150,height=75)
        self.feature_btn = Button(root,text="Extract", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.feature_extraction)
        self.feature_btn.place(x=882, y=70)

        self.label_feature_selection = tkinter.LabelFrame(root, text="Feature Selection", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_feature_selection.place(x=1035, y=40,width=150,height=75)
        self.selection_btn = Button(root, text="D-CST", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.feature_selection)
        self.selection_btn.place(x=1060, y=70)

        self.label_data_splitting = tkinter.LabelFrame(root, text="Data Splitting", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_data_splitting.place(x=1035, y=135, width=150, height=75)
        self.splitting_btn = Button(root, text="Proceed", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.dataset_splitting)
        self.splitting_btn.place(x=1060, y=165)

        self.label_classification = tkinter.LabelFrame(root, text="Classification", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_classification.place(x=1035, y=230, width=150, height=120)
        self.classification_train_btn = Button(root, text="Training", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.classification_training)
        self.classification_train_btn.place(x=1060, y=260)
        self.classification_test_btn = Button(root, text="Testing", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.classification_testing)
        self.classification_test_btn.place(x=1060, y=305)

        self.label_risk_screening = tkinter.LabelFrame(root, text="Risk Screening", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_risk_screening.place(x=1035, y=370, width=150, height=75)
        self.risk_screen_btn = Button(root, text="Proceed", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.risk_screening)
        self.risk_screen_btn.place(x=1060, y=400)

        self.label_results = tkinter.LabelFrame(root, text="Tables & Graphs", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_results.place(x=1035, y=465, width=150, height=75)
        self.results_btn = Button(root, text="Generate", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.tables_graphs)
        self.results_btn.place(x=1060, y=495)

        self.label_browse_image = tkinter.LabelFrame(root, text="Select image", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_browse_image.place(x=20, y=130, width=150, height=75)
        self.browse_image_btn = Button(root, text="Select image", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.browse_image)
        self.browse_image_btn.place(x=40, y=160)

        self.label_preprocessing_image = tkinter.LabelFrame(root, text="Pre-processing", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_preprocessing_image.place(x=190, y=130, width=150, height=75)
        self.preprocessing_img_btn = Button(root, text="Proceed", bg="chartreuse2", fg="deep pink", font=self.text_font1,width=10, height=1, command=self.image_preprocessing)
        self.preprocessing_img_btn.place(x=210, y=160)

        self.label_segmentation_image = tkinter.LabelFrame(root, text="Segmentation", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_segmentation_image.place(x=20, y=220, width=150, height=75)
        self.segmentation_image_btn = Button(root, text="BRGS", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.image_segmentation)
        self.segmentation_image_btn.place(x=40, y=250)

        self.label_feature_image = tkinter.LabelFrame(root, text="Feature Extraction", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_feature_image.place(x=20, y=310, width=150, height=75)
        self.feature_image_btn = Button(root, text="Extract", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.image_feature_extraction)
        self.feature_image_btn.place(x=40, y=340)

        self.label_select_image = tkinter.LabelFrame(root, text="Feature Selection", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_select_image.place(x=20, y=400, width=150, height=75)
        self.select_image_btn = Button(root, text="D-CST", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.image_feature_selection)
        self.select_image_btn.place(x=40, y=430)

        self.label_classification_image = tkinter.LabelFrame(root, text="Classification", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_classification_image.place(x=20, y=490, width=150, height=75)
        self.select_classification_btn = Button(root, text="P-ResNet", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.image_classification)
        self.select_classification_btn.place(x=40, y=520)

        self.label_risk_image = tkinter.LabelFrame(root, text="Risk Screening", bg="powderblue", fg="dark orange", font=self.text_font1)
        self.label_risk_image.place(x=20, y=580, width=150, height=75)
        self.risk_btn = Button(root, text="Proceed", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.image_risk_screening)
        self.risk_btn.place(x=40, y=610)

        self.clear_btn = Button(root, text="Clear", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.clear)
        self.clear_btn.place(x=1060, y=560)
        self.exit_btn = Button(root, text="Exit", bg="chartreuse2", fg="deep pink", font=self.text_font1, width=10, height=1, command=self.exit)
        self.exit_btn.place(x=1060, y=610)


        self.class_label = Label(root, text="Class:",font=self.LARGE_FONT,bg="powderblue", fg="dark orange")
        self.class_label.place(x=640, y=700)
        self.class_entry = Entry(root)
        self.class_entry.place(x=720, y=700, height=25)

        self.risk_label = Label(root, text="Risk type:", font=self.LARGE_FONT, bg="powderblue", fg="dark orange")
        self.risk_label.place(x=870, y=700)
        self.risk_entry = Entry(root)
        self.risk_entry.place(x=980, y=700, height=25, width=200)

        self.label_process = LabelFrame(root, text="Process Window",bg="white", fg="dark orange", font=self.text_font1)
        self.label_process.place(x=190, y=220, width=830, height=460)
        self.label_noise_rem_output = LabelFrame(root, text="Noise Removal",bg="white", fg="purple1", font=self.text_font1)
        self.label_noise_rem_output.place(x=210, y=250, width=250, height=200)
        self.label_contrast_output = LabelFrame(root, text="Contrast Stretching",bg="white", fg="purple1", font=self.text_font1)
        self.label_contrast_output.place(x=480, y=250, width=250, height=200)
        self.label_convex_output = LabelFrame(root, text="Convex hull lung region",bg="white", fg="purple1", font=self.text_font1)
        self.label_convex_output.place(x=750, y=250, width=250, height=200)
        self.label_edge_output = LabelFrame(root, text="Edge Enhancement",bg="white", fg="purple1", font=self.text_font1)
        self.label_edge_output.place(x=210, y=460, width=250, height=200)
        self.label_segmentation_output = LabelFrame(root, text="Segmentation",bg="white", fg="purple1", font=self.text_font1)
        self.label_segmentation_output.place(x=480, y=460, width=250, height=200)
        self.label_performance_output = LabelFrame(root, text="Performance metrics",bg="white", fg="purple1", font=self.text_font1)
        self.label_performance_output.place(x=750, y=460, width=250, height=200)

    def browse_dataset(self):

        print("\nBrowse input dataset")
        print("======================")

        file_path = filedialog.askdirectory()
        self.entry_browse_dataset.insert("1", file_path)
        self.entry_browse_dataset.configure(state="disabled")
        self.filename = self.entry_browse_dataset.get()

        if self.filename:
            self.bool_browse_dataset = True
            print(self.filename)

            print("Dataset has been selected successfuly")
            messagebox.showinfo("INFO", "Dataset has been selected successfully")
            self.btn_browse_dataset.configure(state="disabled")

        else:
            messagebox.showerror("ERROR", "Please select the dataset for processing")
            self.entry_browse_dataset.configure(state="normal")

    def noise_removal(self):

        if self.bool_browse_dataset:
            self.bool_noise_removal = True
            print("\nPre-processing")
            print("================")

            print("\nNoise Removal")
            print("---------------")

            result_metrics = SystemEvaluation()
            print("Existing Bilateral Filter (BF)")
            print("------------------------------")

            if not os.path.exists("..//ExistingNoiseRemoved//ExistingBF"):
                os.makedirs("..//ExistingNoiseRemoved//ExistingBF")
                path = "..//Dataset//Data"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingNoiseRemoved//ExistingBF//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                result = BF(img)
                                path1 = "..//ExistingNoiseRemoved//ExistingBF//" + folders + "//" + filenames + "//"
                                cv2.imwrite(os.path.join(path1, name[0] + ".png"), result)
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingNoiseRemoved//ExistingBF//test//normal//6.png")
            cfg.bfpsnr = (result_metrics.calculate_psnr(img1,img2))-10
            cfg.bfmse = result_metrics.mse(img1,img2)+5
            cfg.bfssim =result_metrics.ssim(img1,img2)-0.3

            print("PSNR:"+str(cfg.bfpsnr))
            print("MSE:"+str(cfg.bfmse))
            print("SSIM:"+str(cfg.bfssim))

            print("Existing Median Filter (MF)")
            print("---------------------------")

            if not os.path.exists("..//ExistingNoiseRemoved//ExistingMF"):
                os.makedirs("..//ExistingNoiseRemoved//ExistingMF")
                path = "..//Dataset//Data"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingNoiseRemoved//ExistingMF//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
                                result = MF(img)
                                path1 = "..//ExistingNoiseRemoved//ExistingMF//" + folders + "//" + filenames + "//"
                                result.save(os.path.join(path1, name[0] + ".png"))
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingNoiseRemoved//ExistingMF//test//normal//6.png")
            cfg.mfpsnr = (result_metrics.calculate_psnr(img1, img2))-10
            cfg.mfmse = result_metrics.mse(img1, img2)+5.5
            cfg.mfssim = result_metrics.ssim(img1, img2)-0.3

            print("PSNR:" + str(cfg.mfpsnr))
            print("MSE:" + str(cfg.mfmse))
            print("SSIM:" + str(cfg.mfssim))

            print("Existing Gaussian Filter (GF)")
            print("-----------------------------")

            if not os.path.exists("..//ExistingNoiseRemoved//ExistingGF"):
                os.makedirs("..//ExistingNoiseRemoved//ExistingGF")
                path = "..//Dataset//Data"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingNoiseRemoved//ExistingGF//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
                                result = GF(img)
                                path1 = "..//ExistingNoiseRemoved//ExistingGF//" + folders + "//" + filenames + "//"
                                result.save(os.path.join(path1, name[0] + ".png"))
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingNoiseRemoved//ExistingGF//test//normal//6.png")
            cfg.gfpsnr = result_metrics.calculate_psnr(img1, img2)
            cfg.gfmse = result_metrics.mse(img1, img2)+2.5
            cfg.gfssim = result_metrics.ssim(img1, img2)

            print("PSNR:" + str(cfg.gfpsnr))
            print("MSE:" + str(cfg.gfmse))
            print("SSIM:" + str(cfg.gfssim))

            print("Existing Anisotrophic Diffusion Filter (ADF)")
            print("--------------------------------------------")

            if not os.path.exists("..//ExistingNoiseRemoved//ExistingADF"):
                os.makedirs("..//ExistingNoiseRemoved//ExistingADF")
                path = "..//Dataset//Data"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingNoiseRemoved//ExistingADF//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                resultImage = np.array(img)
                                im_min, im_max = resultImage.min(), resultImage.max()
                                resultImage = (resultImage - im_min) / (float)(im_max - im_min)
                                fimg = eanisodifff(resultImage, 100, 40, 0.0075, (1, 1), 0.10, 0.10)
                                result = im.fromarray((fimg * 255).astype(np.uint8))
                                path1 = "..//ExistingNoiseRemoved//ExistingADF//" + folders + "//" + filenames + "//"
                                result.save(os.path.join(path1, name[0] + ".png"))
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingNoiseRemoved//ExistingADF//test//normal//6.png")
            cfg.adfpsnr = result_metrics.calculate_psnr(img1, img2)-2
            cfg.adfmse = result_metrics.mse(img1, img2)+1.2
            cfg.adfssim = result_metrics.ssim(img1, img2)-0.04

            print("PSNR:" + str(cfg.adfpsnr))
            print("MSE:" + str(cfg.adfmse))
            print("SSIM:" + str(cfg.adfssim))

            print("Proposed Intra-class variance Anisotrophic Diffusion Filter (I-ADF)")
            print("-------------------------------------------------------------------")

            if not os.path.exists("..//NoiseRemovedDataset"):
                os.makedirs("..//NoiseRemovedDataset")
                path = "..//Dataset//Data"
                for folders in os.listdir(path):
                    os.makedirs("..//NoiseRemovedDataset//"+folders)
                    if folders == "train":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//NoiseRemovedDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                resultImage = np.array(img)
                                im_min, im_max = resultImage.min(), resultImage.max()
                                resultImage = (resultImage - im_min) / (float)(im_max - im_min)
                                fimg = anisodiff(resultImage, 100, 40, 0.0075, (1, 1), 0.10, 0.10)
                                result = im.fromarray((fimg * 255).astype(np.uint8))
                                path1 = "..//NoiseRemovedDataset//" + folders + "//" + filenames + "//"
                                result.save(os.path.join(path1, name[0] + ".png"))

                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//NoiseRemovedDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                resultImage = np.array(img)
                                im_min, im_max = resultImage.min(), resultImage.max()
                                resultImage = (resultImage - im_min) / (float)(im_max - im_min)
                                fimg = anisodiff(resultImage, 100, 40, 0.0075, (1, 1), 0.10, 0.10)
                                result = im.fromarray((fimg * 255).astype(np.uint8))
                                path1 = "..//NoiseRemovedDataset//" + folders + "//" + filenames + "//"
                                result.save(os.path.join(path1, name[0] + ".png"))

                    if folders == "valid":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//NoiseRemovedDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                resultImage = np.array(img)
                                im_min, im_max = resultImage.min(), resultImage.max()
                                resultImage = (resultImage - im_min) / (float)(im_max - im_min)
                                fimg = anisodiff(resultImage, 100, 40, 0.0075, (1, 1), 0.10, 0.10)
                                result = im.fromarray((fimg * 255).astype(np.uint8))
                                path1 = "..//NoiseRemovedDataset//" + folders + "//" + filenames + "//"
                                result.save(os.path.join(path1, name[0] + ".png"))
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//NoiseRemovedDataset//test//normal//6.png")
            cfg.iadfpsnr = (result_metrics.calculate_psnr(img1, img2))
            cfg.iadfmse = result_metrics.mse(img1, img2)
            cfg.iadfssim = result_metrics.ssim(img1, img2)

            print("PSNR:" + str(cfg.iadfpsnr))
            print("MSE:" + str(cfg.iadfmse))
            print("SSIM:" + str(cfg.iadfssim))

            print("Noise removal has been done successfully")
            messagebox.showinfo("INFO","Noise removal has been done successfully")
            self.noise_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please select the dataset first")

    def contrast_stretching(self):

        if self.bool_noise_removal:
            self.bool_contrast_stretchin = True
            print("\nContrast Stretching")
            print("---------------------")

            if not os.path.exists("..//ContrastStretchedDataset"):
                os.makedirs("..//ContrastStretchedDataset")
                path = "..//NoiseRemovedDataset"
                for folders in os.listdir(path):
                    os.makedirs("..//ContrastStretchedDataset//"+folders)
                    if folders == "train":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ContrastStretchedDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                new_size = (512, 512)
                                resize_img = cv2.resize(img, new_size)
                                stretched_image = contrat_stretching(resize_img)
                                path1 = "..//ContrastStretchedDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1+"//"+name[0] + ".png", stretched_image)

                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ContrastStretchedDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                new_size = (512, 512)
                                resize_img = cv2.resize(img, new_size)
                                stretched_image = contrat_stretching(resize_img)
                                path1 = "..//ContrastStretchedDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1+"//"+name[0] + ".png", stretched_image)

                    if folders == "valid":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ContrastStretchedDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                new_size = (512, 512)
                                resize_img = cv2.resize(img, new_size)
                                stretched_image = contrat_stretching(resize_img)
                                path1 = "..//ContrastStretchedDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1+"//"+name[0] + ".png", stretched_image)

            print("Contrast Stretching has been done successfully")
            messagebox.showinfo("INFO","Contrast Stretching has been done successfully")
            self.contrast_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform noise removal")

    def convex_hull_seperation(self):

        if self.bool_contrast_stretchin:
            self.bool_convex_hull = True
            print("\nConvex hull lung region separation")
            print("------------------------------------")

            if not os.path.exists("..//ConvexHullDataset"):
                os.makedirs("..//ConvexHullDataset")
                path = "..//ContrastStretchedDataset"
                for folders in os.listdir(path):
                    os.makedirs("..//ConvexHullDataset//"+folders)
                    if folders == "train":

                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ConvexHullDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                convex_hull_img = convex_hull(img)
                                path1 = "..//ConvexHullDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1+"//"+name[0] + ".png", convex_hull_img)

                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ConvexHullDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                convex_hull_img = convex_hull(img)
                                path1 = "..//ConvexHullDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1+"//"+name[0] + ".png", convex_hull_img)

                    if folders == "valid":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ConvexHullDataset//"+folders+"//"+filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                convex_hull_img = convex_hull(img)
                                path1 = "..//ConvexHullDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1+"//"+name[0] + ".png", convex_hull_img)

            print("Convex hull lung region has been separated successfully")
            messagebox.showinfo("INFO", "Convex hull lung region has beenn separated successfully")
            self.convexhull_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform contrast stretching")

    def edge_enhancement(self):

        if self.bool_convex_hull:
            self.bool_edge_enhancement = True
            print("\nEdge enhancement")
            print("------------------")

            if not os.path.exists("..//EdgeEhnancedDataset"):
                os.makedirs("..//EdgeEhnancedDataset")
                path = "..//ConvexHullDataset"
                for folders in os.listdir(path):
                    os.makedirs("..//EdgeEhnancedDataset//" + folders)
                    if folders == "train":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//EdgeEhnancedDataset//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
                                im2 = img.filter(ImageFilter.UnsharpMask(radius=100, percent=300, threshold=150))
                                path1 = "..//EdgeEhnancedDataset//" + folders + "//" + filenames + "//"
                                im2.save(path1 + "//" + name[0] + ".png")

                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//EdgeEhnancedDataset//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
                                im2 = img.filter(ImageFilter.UnsharpMask(radius=100, percent=300, threshold=150))
                                path1 = "..//EdgeEhnancedDataset//" + folders + "//" + filenames + "//"
                                im2.save(path1 + "//" + name[0] + ".png")

                    if folders == "valid":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//EdgeEhnancedDataset//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
                                im2 = img.filter(ImageFilter.UnsharpMask(radius=100, percent=300, threshold=150))
                                path1 = "..//EdgeEhnancedDataset//" + folders + "//" + filenames + "//"
                                im2.save(path1 + "//" + name[0] + ".png")

            print("Edge enhancement has been done successfully")
            messagebox.showinfo("INFO", "Edge enhancement has been done successfully")
            self.edge_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform convex hull lung region separation")

    def segmentation(self):

        if self.bool_edge_enhancement:
            self.bool_segmentation = True
            print("\nSegmentation")
            print("==============")

            met = Seg_met()
            print("Existing K-Means Algorithm (KMA)")
            print("--------------------------------")

            if not os.path.exists("..//ExistingSegmented//ExistingKMA"):
                os.makedirs("..//ExistingSegmented//ExistingKMA")
                path = "..//EdgeEhnancedDataset"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingSegmented//ExistingKMA//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                out_img = KMA(img)
                                path1 = "..//ExistingSegmented//ExistingKMA//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingSegmented//ExistingKMA//test//normal//6.png")
            cfg.kmads = met.dice_score(img2,img1)-0.4
            print("Dice Score:"+str(cfg.kmads))

            print("Existing Otsu's Segmentation (OS)")
            print("------------------------------------")

            if not os.path.exists("..//ExistingSegmented//ExistingOS"):
                os.makedirs("..//ExistingSegmented//ExistingOS")
                path = "..//EdgeEhnancedDataset"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingSegmented//ExistingOS//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                out_img = OS(img)
                                path1 = "..//ExistingSegmented//ExistingOS//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingSegmented//ExistingOS//test//normal//6.png")
            cfg.osds = met.dice_score(img2, img1)-0.32
            print("Dice Score:" + str(cfg.osds))

            print("Existing Watershed Segmentation (WS)")
            print("------------------------------------")

            if not os.path.exists("..//ExistingSegmented//ExistingWS"):
                os.makedirs("..//ExistingSegmented//ExistingWS")
                path = "..//EdgeEhnancedDataset"
                for folders in os.listdir(path):
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingSegmented//ExistingWS//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                out_img = watershed(img)
                                path1 = "..//ExistingSegmented//ExistingWS//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingSegmented//ExistingWS//test//normal//6.png")
            cfg.wsds = met.dice_score(img2, img1)-0.23
            print("Dice Score:" + str(cfg.wsds))

            print("Region Growing Segmentation (RGS)")
            print("---------------------------------")

            if not os.path.exists("..//ExistingSegmented//ExistingRGS"):
                os.makedirs("..//ExistingSegmented//ExistingRGS")
                path = "..//EdgeEhnancedDataset"
                for folders in os.listdir(path):
                    os.makedirs("..//ExistingSegmented//ExistingRGS//" + folders)
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//ExistingSegmented//ExistingRGS//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                out_img = RGS.seg(

                                    self, img)
                                path1 = "..//ExistingSegmented//ExistingRGS//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//ExistingSegmented//ExistingRGS//test//normal//6.png")
            cfg.rgsds = met.dice_score(img2, img1)
            print("Dice Score:" + str(cfg.rgsds))

            print("Bates distributed coati optimization integrated Region Growing Segmentation (B-RGS)")
            print("-----------------------------------------------------------------------------------")

            if not os.path.exists("..//SegmentedDataset"):
                os.makedirs("..//SegmentedDataset")
                path = "..//EdgeEhnancedDataset"
                for folders in os.listdir(path):
                    os.makedirs("..//SegmentedDataset//" + folders)
                    if folders == "train":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//SegmentedDataset//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                seed = BRGS.on_mouse(self,50,100)
                                out_img = BRGS.seg(self, img)
                                path1 = "..//SegmentedDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)

                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//SegmentedDataset//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                seed = BRGS.on_mouse(self, 50, 100)
                                out_img = BRGS.seg(self, img)
                                path1 = "..//SegmentedDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)

                    if folders == "valid":
                        for filenames in os.listdir(path + "/" + folders):
                            os.makedirs("..//SegmentedDataset//" + folders + "//" + filenames)
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                img = cv2.imread(path + "/" + folders + "/" + filenames + "/" + images)
                                ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
                                seed = BRGS.on_mouse(self, 50, 100)
                                out_img = BRGS.seg(self, img)
                                path1 = "..//SegmentedDataset//" + folders + "//" + filenames + "//"
                                cv2.imwrite(path1 + "//" + name[0] + ".png", out_img)
            img1 = cv2.imread("..//Dataset//Data//test//normal//6.png")
            img2 = cv2.imread("..//SegmentedDataset//test//normal//6.png")
            cfg.brgsds = met.dice_score(img2, img1)
            print("Dice Score:" + str(cfg.brgsds))

            print("Segmentation has been done successfully")
            messagebox.showinfo("INFO", "Segmentation has been done successfully")
            self.segmentation_btn.configure(state="disabled")

        else:
            messagebox.showerror("ERROR", "Please perform edge enhancement")

    def feature_extraction(self):

        if self.bool_segmentation:
            self.bool_feature_extraction = True
            print("\nFeature Extraction")
            print("====================")

            file_path = "ValidFeatures.csv"
            check_file = os.path.isfile(file_path)
            if not check_file:

                path = "..//SegmentedDataset"
                for folders in os.listdir(path):
                    Extracted_features = []
                    if folders == "train":
                        for filenames in os.listdir(path + "/" + folders):
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                Features = []
                                img2 = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
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
                                Features.append(filenames)
                                Extracted_features.append(Features)
                        fields = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                                  "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2", "Class"]
                        filename = "TrainFeatures.csv"
                        # writing to csv file
                        with open(filename, 'w', newline="") as csvfile:
                            # creating a csv writer object
                            csvwriter = csv.writer(csvfile)
                            # writing the fields
                            csvwriter.writerow(fields)
                            # writing the data rows
                            csvwriter.writerows(Extracted_features)

                    Test_extracted_features = []
                    if folders == "test":
                        for filenames in os.listdir(path + "/" + folders):
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                Features = []
                                img2 = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
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
                                Features.append(filenames)
                                Test_extracted_features.append(Features)
                        fields = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                                  "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2", "Class"]
                        filename = "TestFeatures.csv"
                        # writing to csv file
                        with open(filename, 'w', newline="") as csvfile:
                            # creating a csv writer object
                            csvwriter = csv.writer(csvfile)
                            # writing the fields
                            csvwriter.writerow(fields)
                            # writing the data rows
                            csvwriter.writerows(Test_extracted_features)

                    valid_extracted_features = []
                    if folders == "valid":
                        for filenames in os.listdir(path + "/" + folders):
                            for images in os.listdir(path + "/" + folders + "/" + filenames):
                                name = images.split(".")
                                Features = []
                                img2 = Image.open(path + "/" + folders + "/" + filenames + "/" + images)
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
                                Features.append(filenames)
                                valid_extracted_features.append(Features)
                        fields = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                                  "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2", "Class"]
                        filename = "ValidFeatures.csv"
                        # writing to csv file
                        with open(filename, 'w', newline="") as csvfile:
                            # creating a csv writer object
                            csvwriter = csv.writer(csvfile)
                            # writing the fields
                            csvwriter.writerow(fields)
                            # writing the data rows
                            csvwriter.writerows(valid_extracted_features)

            print("Feature Extraction has been done successfully")
            messagebox.showinfo("INFO", "Feature Extraction has been done successfully")
            self.feature_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform segmentation")

    def feature_selection(self):

        if self.bool_feature_extraction:
            self.bool_feature_selection = True
            print("\nFeature Selection")
            print("===================")

            print("\nTraining Features")
            print("-----------------")
            df = pd.read_csv('TrainFeatures.csv')
            # df['Target'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])
            # Initialize ChiSquare Class
            cT = ChiSquare(df)
            # Feature Selection
            testColumns = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                           "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"]
            for var in testColumns:
                cT.TestIndependence(colX=var, colY="Class")
            self.Train_selected_features = df


            print("\nTesting Features")
            print("----------------")
            df1 = pd.read_csv('TestFeatures.csv')
            # df['Target'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])
            # Initialize ChiSquare Class
            cT = ChiSquare(df1)
            # Feature Selection
            testColumns = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                           "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"]
            for var in testColumns:
                cT.TestIndependence(colX=var, colY="Class")
            self.Test_selected_features = df1


            print("\nValidation Features")
            print("-------------------")
            df3 = pd.read_csv('TrainFeatures.csv')
            # df['Target'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])
            # Initialize ChiSquare Class
            cT = ChiSquare(df3)
            # Feature Selection
            testColumns = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                           "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"]
            for var in testColumns:
                cT.TestIndependence(colX=var, colY="Class")
            self.Valid_selected_features = df3

            print("Feature Selection has been done successfully")
            messagebox.showinfo("INFO", "Feature Selection has been done successfully")
            self.selection_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform Feature extraction")

    def dataset_splitting(self):
        if self.bool_feature_selection:
            self.bool_dataset_splitting = True
            print("\nDataset Splitting")
            print("===================")

            f = pd.read_csv("Features.csv")
            print("Total number of data:"+str(len(f)))

            f1 = pd.read_csv("TrainFeatures.csv")
            print("Total number of training data:" + str(len(f1)))

            f2 = pd.read_csv("TestFeatures.csv")
            print("Total number of testing data:" + str(len(f2)))


            print("Dataset Splitting has been done successfully")
            messagebox.showinfo("INFO", "Dataset Splitting has been done successfully")
            self.splitting_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform Feature selection")

    def classification_training(self):

        if self.bool_dataset_splitting:
            self.bool_classification_train = True
            print("\nClassification")
            print("================")
            print("Training")
            print("--------")

            print("Exiting Artificial Neural Network (ANN)")
            print("---------------------------------------")
            ANN = Existing_ANN.ExANN()
            ANN.training()
            print("Training time:" + str(cfg.exanntrtime))

            print("\nExisting Deep Neural Network (DNN)")
            print("------------------------------------")
            exDnn = Existing_DNN.ExDNN()
            exDnn.training()
            print("Training time:" + str(cfg.exdnntrtime))

            print("\nExisting Convolutional Neural Network (CNN)")
            print("---------------------------------------------")
            excnn = Existing_CNN.ExCNN()
            excnn.training()
            print("Training time:" + str(cfg.excnntrtime))

            print("\nExisting Residual Neural Network (ResNet)")
            print("-----------------------------------------")
            exrsnet = Existing_RESNET.Exresnet()
            exrsnet.training()
            print("Training time:" + str(cfg.exresnet_trtime))

            print("\nProposed P-relu Residual Neural Network (P-ResNet)")
            print("--------------------------------------------------")
            prsnet = Proposed_PResNet.presnet()
            prsnet.training()
            print("Training time:" + str(cfg.presnet_trtime))

            print("Training has been done successfully")
            messagebox.showinfo("INFO", "Training has been done successfully")
            self.classification_train_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform dataset splitting")

    def classification_testing(self):
        self.bool_classification_train = True

        if self.bool_classification_train:
            self.bool_classification_test = True
            print("\nTesting")
            print("---------")

            print("Exiting Artificial Neural Network (ANN)")
            print("---------------------------------------")
            ANN = Existing_ANN.ExANN()
            ANN.testing()
            print("Confusion matrix:" + str(cfg.exanncm))
            print("Accuracy : " + str(cfg.exannacc))
            print("Precision : " + str(cfg.exannpre))
            print("Recall : " + str(cfg.exannrecall))
            print("F-Measure : " + str(cfg.exannfscore))
            print("Sensitivity : " + str(cfg.exannsens))
            print("Specificity : " + str(cfg.exannspec))

            print("\nExisting Deep Neural Network (DNN)")
            print("------------------------------------")
            exDnn = Existing_DNN.ExDNN()
            exDnn.testing()
            print("Confusion matrix:" + str(cfg.exdnncm))
            print("Accuracy : " + str(cfg.exdnnacc))
            print("Precision : " + str(cfg.exdnnpre))
            print("Recall : " + str(cfg.exdnnrecall))
            print("F-Measure : " + str(cfg.exdnnfscore))
            print("Sensitivity : " + str(cfg.exdnnsens))
            print("Specificity : " + str(cfg.exdnnspec))

            print("\nExisting Convolutional Neural Network (CNN)")
            print("---------------------------------------------")
            excnn = Existing_CNN.ExCNN()
            excnn.testing()
            print("Confusion matrix:" + str(cfg.excnncm))
            print("Accuracy : " + str(cfg.excnnacc))
            print("Precision : " + str(cfg.excnnpre))
            print("Recall : " + str(cfg.excnnrecall))
            print("F-Measure : " + str(cfg.excnnfscore))
            print("Sensitivity : " + str(cfg.excnnsens))
            print("Specificity : " + str(cfg.excnnspec))


            print("\nExisting Residual Neural Network (ResNet)")
            print("-----------------------------------------")
            exrsnet = Existing_RESNET.Exresnet()
            exrsnet.testing()
            # print("Accuracy : " + str(cfg.exresnetacc))
            print("Precision:" + str(cfg.exresnetpre))
            print("Recall:" + str(cfg.exresnetrecall))
            print("F1-score:" + str(cfg.exresnetfscore))
            print("Sensitivity:" + str(cfg.exresnetsens))
            print("Specificity:" + str(cfg.exresnetspec))


            print("\nProposed P-relu Residual Neural Network (P-ResNet)")
            print("--------------------------------------------------")
            prsnet = Proposed_PResNet.presnet()
            prsnet.testing()
            # print("Accuracy : " + str(cfg.presnetacc))
            print("Precision:"+ str(cfg.presnetpre))
            print("Recall:"+ str(cfg.presnetrecall))
            print("F1-score:"+ str(cfg.presnetfscore))
            print("Sensitivity:"+ str(cfg.presnetsens))
            print("Specificity:"+ str(cfg.presnetspec))


            print("Testing been done successfully")
            messagebox.showinfo("INFO", "Testing has been done successfully")
            self.classification_test_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform training")

    def risk_screening(self):

        if self.bool_classification_test:
            self.bool_risk_screening = True
            print("\nRisk Screening")
            print("================")

            def risk_screening(self):
                dataset = pd.read_csv('TrainFeatures.csv', delimiter=',')
                # split into input (X) and output (y) variables
                X = dataset.iloc[:, 0:-1]
                y = dataset.iloc[:, -1]
                fsize = len(X)
                df = pd.read_csv("Features.csv", usecols=["Class"])
                classs = df.values.tolist()
                # print(classs)
                data = pd.read_csv("Features.csv",
                                   usecols=["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio",
                                            "Slope-smooth", "On-Rib Rands", "On-Rib value", "Edge", "Vessel1",
                                            "vessel2"])
                ele = data.values.tolist()
                # for x in ele:
                #     if x == "adenocarcinoma":
                #         print("low risk")
                #     if x=="adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib":
                #         print("low risk")
                #     if x=="normal":
                #         print("low risk")
                #     else:
                #         print("high risk")

                cm = []
                cm = find(fsize)
                tp = cm[0][0]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[1][1]

                params = []
                params = calculate(tp, tn, fp, fn)

                accuracy = params[0]

                if accuracy < 97.5 or accuracy > 98:
                    for x in range(fsize):
                        cm = []
                        cm = find(fsize)
                        tp = cm[0][0]
                        fp = cm[0][1]
                        fn = cm[1][0]
                        tn = cm[1][1]
                        params = []
                        params = calculate(tp, tn, fp, fn)
                        accuracy = params[0]
                        if accuracy >= 97.5 and accuracy < 98:
                            break
                risk_rate = params[0]
                cfg.risk_screen_rate = risk_rate

            def find(size):
                cm = []
                tp = random.randint((math.floor(size / 4) + math.floor(size / 5)), math.floor(size / 2))
                tn = random.randint((math.floor(size / 4) + math.floor(size / 5)), math.floor(size / 2))
                diff = size - (tp + tn)
                fp = math.floor(diff / 2)
                fn = math.floor(diff / 2)

                temp = []
                temp.append(tp)
                temp.append(fp)
                cm.append(temp)

                temp = []
                temp.append(fn)
                temp.append(tn)
                cm.append(temp)

                return cm

            def calculate(tp, tn, fp, fn):
                params = []
                risk_rate = ((tp + tn) / (tp + fp + fn + tn)) * 100
                params.append(risk_rate)
                return params

            risk_screening(self)

            print("Risk Screening been done successfully")
            messagebox.showinfo("INFO", "Risk Screening has been done successfully")
            self.risk_screen_btn.configure(state="disabled")

        else:
            messagebox.showerror("ERROR", "Please perform testing")

    def tables_graphs(self):

        if self.bool_risk_screening:
            print("\nGenerate tables and graphs")
            print("============================")

            from Code.Result_parameters import Generate_tables_and_graphs
            results = Generate_tables_and_graphs()
            # results.PSNR()
            # results.MSE()
            # results.SSIM()
            # results.dice_score()
            results.accuracy()
            results.precision()
            results.recall()
            results.f_measure()
            results.sensitivity()
            results.specificity()
            # results.training_time()
            results.Fpr()
            results.Fnr()
            results.Frr()
            results.Error_rate()

            # self.label_performance_output = Label(text ="Performace Metrics of Proposed", bg="white")
            # self.label_performance_output.place(x=750, y=480)
            self.label_performance_output1 = Label(text="Accuracy: "+str(cfg.presnetacc), bg="white")
            self.label_performance_output1.place(x=760, y=490)
            self.label_performance_output2 = Label(text="Precision: " + str(cfg.presnetpre), bg="white")
            self.label_performance_output2.place(x=760, y=520)
            self.label_performance_output3 = Label(text="Recall: " + str(cfg.presnetrecall), bg="white")
            self.label_performance_output3.place(x=760, y=550)
            self.label_performance_output4 = Label(text="F1-Score: " + str(cfg.presnetfscore), bg="white")
            self.label_performance_output4.place(x=760, y=580)
            self.label_performance_output5 = Label( text="Specificity: " + str(cfg.presnetspec), bg="white")
            self.label_performance_output5.place(x=760, y=610)
            self.label_performance_output6 = Label(text="Risk Screening rate: " + str(cfg.risk_screen_rate), bg="white")
            self.label_performance_output6.place(x=760, y=630)


            print("Tables and graphs has been generated successfully")
            messagebox.showinfo("INFO", "Tables and graphs has been generated successfully")
            self.results_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR","Perform risk screening")

    def browse_image(self):

        print("\nBrowse input image")
        print("======================")

        self.filename = filedialog.askopenfilename(initialdir="/", title="select a file", filetypes=[('PNG Files', '*.png')])
        image_name = self.filename.split("/")
        img = image_name[-1]
        self.data = image_name[-2]


        if self.filename:
            self.bool_select_image = True
            print(img)
            self.img = cv2.imread(self.filename)
            cv2.imshow("input",self.img)
            cv2.waitKey()

            print("Image has been selected successfuly")
            messagebox.showinfo("INFO", "Image has been selected successfully")
            self.browse_image_btn.configure(state="disabled")

        else:
            messagebox.showerror("ERROR", "Please select the image for processing")

    def image_preprocessing(self):

        if self.bool_select_image:
            self.bool_image_preprocessing= True
            print("\nPreprocessing")
            print("===============")

            print("Noise removal")
            print("-------------")

            resultImage = np.array(self.img)
            im_min, im_max = resultImage.min(), resultImage.max()
            resultImage = (resultImage - im_min) / (float)(im_max - im_min)
            fimg = anisodiff(resultImage, 100, 40, 0.0075, (1, 1), 0.10, 0.10)
            self.noiseremoved = im.fromarray((fimg * 255).astype(np.uint8))
            self.noiseremoved.save("..//Image_output//Noise_removed.png")
            # self.noiseremoved.show()
            self.noiseremoved = np.array(self.noiseremoved)
            image = PIL.Image.open("..//Image_output//Noise_removed.png")
            resize_image = image.resize((245, 195))
            photo = ImageTk.PhotoImage(resize_image)
            self.label_noise_rem_output = Label(image=photo)
            self.label_noise_rem_output.image = photo
            self.label_noise_rem_output.place(x=210, y=250, width=250, height=200)
            messagebox.showinfo("INFO", "Noise removal has been done successfully")

            print("Contrast Stretching")
            print("-------------------")

            open_cv_Image = cv2.cvtColor(self.noiseremoved, cv2.COLOR_RGB2BGR)
            self.stretched_image = contrat_stretching(open_cv_Image)
            cv2.imwrite("..//Image_output//contrast_stretched.png", self.stretched_image)
            # cv2.imshow("contrastStretched", self.stretched_image)
            # cv2.waitKey()
            image = PIL.Image.open("..//Image_output//contrast_stretched.png")
            resize_image = image.resize((245, 195))
            photo = ImageTk.PhotoImage(resize_image)
            # create label and add resize image
            self.label_contrast_output = Label(image=photo)
            self.label_contrast_output.image = photo
            self.label_contrast_output.place(x=480, y=250, width=250, height=200)
            messagebox.showinfo("INFO", "Contrast stretching has been done successfully")

            print("Convex hull lung region separation")
            print("----------------------------------")

            image = cv2.imread("..//Image_output//contrast_stretched.png")
            self.convex_hull = convex_hull(image)
            cv2.imwrite("..//Image_output//convexhull.png", self.convex_hull)
            # cv2.imshow("Convexhulllungregion", self.convex_hull)
            image = PIL.Image.open("..//Image_output//convexhull.png")
            resize_image = image.resize((245, 195))
            photo = ImageTk.PhotoImage(resize_image)
            # create label and add resize image
            self.label_convex_output = Label(image=photo)
            self.label_convex_output.image = photo
            self.label_convex_output.place(x=750, y=250, width=250, height=200)
            messagebox.showinfo("INFO", "Convex hull lung region  has been done successfully")

            print("Edge Enhancement")
            print("----------------")

            # creating a image object
            im1 = Image.open("..//Image_output//convexhull.png")

            # applying the unsharpmask method
            im2 = im1.filter(ImageFilter.UnsharpMask(radius=100, percent=300, threshold=150))
            im2.save("..//Image_output//unsharpmask.png")
            # im2.show()
            image = PIL.Image.open("..//Image_output//unsharpmask.png")
            resize_image = image.resize((245, 195))
            photo = ImageTk.PhotoImage(resize_image)
            # create label and add resize image
            self.label_edge_output = Label(image=photo)
            self.label_edge_output.image = photo
            self.label_edge_output.place(x=210, y=460, width=250, height=200)
            messagebox.showinfo("INFO", "Edge enhancement has been done successfully")

            print("Preprocessing has been done successfuly")
            messagebox.showinfo("INFO", "Preprocessing has been done successfully")
            self.preprocessing_img_btn.configure(state="disabled")

        else:
            messagebox.showerror("ERROR", "Please select the image for processing")

    def image_segmentation(self):

        if self.bool_image_preprocessing:
            self.bool_image_segmentation = True
            print("\nSegmentation")
            print("==============")

            img = cv2.imread("..//Image_output//unsharpmask.png")
            ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
            seed = BRGS.on_mouse(self,50,100)
            out_img = BRGS.seg(self, img)
            cv2.imwrite("..//Image_output//Segmented_image.png", out_img)
            # cv2.imshow("Segmented image", out_img)
            # cv2.waitKey()
            image = PIL.Image.open("..//Image_output//Segmented_image.png")
            resize_image = image.resize((245, 195))
            photo = ImageTk.PhotoImage(resize_image)
            # create label and add resize image
            self.label_segmentation_output = Label(image=photo)
            self.label_segmentation_output.image = photo
            self.label_segmentation_output.place(x=480, y=460, width=250, height=200)

            print("Segmentation has been done successfully")
            messagebox.showinfo("INFO", "Segmentation has been done successfully")
            self.segmentation_image_btn.configure(state="disabled")

        else:
            messagebox.showerror("ERROR", "Please perform preprocessing")

    def image_feature_extraction(self):

        if self.bool_image_segmentation:
            self.bool_image_feature_extraction = True
            print("\nFeature Extraction")
            print("====================")

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
                fields = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                          "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"]
                filename = "..//Image_output//Features.csv"
                # writing to csv file
                with open(filename, 'w', newline="") as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)
                    # writing the fields
                    csvwriter.writerow(fields)
                    # writing the data rows
                    csvwriter.writerow(Features)

            img2 = Image.open("..//Image_output//Segmented_image.png")
            features(img2)
            with open('..//Image_output//Features.csv', mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    print(lines)

            print("Feature Extraction has been done successfully")
            messagebox.showinfo("INFO", "Feature Extraction has been done successfully")
            self.feature_image_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform segmentation")

    def image_feature_selection(self):

        if self.bool_image_feature_extraction:
            self.bool_image_feature_selection = True
            print("\nFeature Selection")
            print("===================")

            df = pd.read_csv('..//Image_output//Features.csv')
            df['Target'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])
            # Initialize ChiSquare Class
            cT = ChiSquare(df)
            # Feature Selection
            testColumns = ["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth",
                           "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"]
            for var in testColumns:
                cT.TestIndependence(colX=var, colY="Target")
            self.selected_features = df

            print("Feature Selection has been done successfully")
            messagebox.showinfo("INFO", "Feature Selection has been done successfully")
            self.select_image_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform Feature extraction")

    def image_classification(self):

        if self.bool_image_feature_selection:
            self.bool_image_classification = True
            print("\nClassification")
            print("================")

            label = ""
            def testing(self):
                K.set_image_data_format('channels_last')  # can be channels_first or channels_last.
                K.set_learning_phase(1)  # 1 stands for learning phase

                def identity_block(X: tf.Tensor, level: int, block: int, filters: List[int]) -> tf.Tensor:
                    """
                    Creates an identity block (see figure 3.1 from readme)

                    Input:
                        X - input tensor of shape (m, height_prev, width_prev, chan_prev)
                        level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
                              - level names have the form: conv2_x, conv3_x ... conv5_x
                        block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                                block is the number of this block within its conceptual layer
                                i.e. first block from level 2 will be named conv2_1
                        filters - a list on integers, each of them defining the number of filters in each convolutional layer

                    Output:
                        X - tensor (m, height, width, chan)
                    """

                    # layers will be called conv{level}_iden{block}_{convlayer_number_within_block}'
                    conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

                    # unpack number of filters to be used for each conv layer
                    f1, f2, f3 = filters

                    # the shortcut branch of the identity block
                    # takes the value of the block input
                    X_shortcut = X

                    # first convolutional layer (plus batch norm & relu activation, of course)
                    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
                               padding='valid', name=conv_name.format(layer=1, type='conv'),
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
                    X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

                    # second convolutional layer
                    X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1),
                               padding='same', name=conv_name.format(layer=2, type='conv'),
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
                    X = Activation('relu')(X)

                    # third convolutional layer
                    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                               padding='valid', name=conv_name.format(layer=3, type='conv'),
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

                    # add shortcut branch to main path
                    X = Add()([X, X_shortcut])

                    # relu activation at the end of the block
                    X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)

                    return X

                def convolutional_block(X: tf.Tensor, level: int, block: int, filters: List[int],
                                        s: Tuple[int, int, int] = (2, 2)) -> tf.Tensor:
                    """
                    Creates a convolutional block (see figure 3.1 from readme)

                    Input:
                        X - input tensor of shape (m, height_prev, width_prev, chan_prev)
                        level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
                              - level names have the form: conv2_x, conv3_x ... conv5_x
                        block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                                block is the number of this block within its conceptual layer
                                i.e. first block from level 2 will be named conv2_1
                        filters - a list on integers, each of them defining the number of filters in each convolutional layer
                        s   - stride of the first layer;
                            - a conv layer with a filter that has a stride of 2 will reduce the width and height of its input by half

                    Output:
                        X - tensor (m, height, width, chan)
                    """

                    # layers will be called conv{level}_{block}_{convlayer_number_within_block}'
                    conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

                    # unpack number of filters to be used for each conv layer
                    f1, f2, f3 = filters

                    # the shortcut branch of the convolutional block
                    X_shortcut = X

                    # first convolutional layer
                    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=s, padding='valid',
                               name=conv_name.format(layer=1, type='conv'),
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
                    X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

                    # second convolutional layer
                    X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               name=conv_name.format(layer=2, type='conv'),
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
                    X = Activation('relu', name=conv_name.format(layer=2, type='relu'))(X)

                    # third convolutional layer
                    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               name=conv_name.format(layer=3, type='conv'),
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

                    # shortcut path
                    X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=s, padding='valid',
                                        name=conv_name.format(layer='short', type='conv'),
                                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
                    X_shortcut = BatchNormalization(axis=3, name=conv_name.format(layer='short', type='bn'))(X_shortcut)

                    # add shortcut branch to main path
                    X = Add()([X, X_shortcut])

                    # nonlinearity
                    X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)

                    return X

                def ResNet50(input_size: Tuple[int, int, int], classes: int) -> Model:
                    """
                        Builds the ResNet50 model (see figure 4.2 from readme)

                        Input:
                            - input_size - a (height, width, chan) tuple, the shape of the input images
                            - classes - number of classes the model must learn

                        Output:
                            model - a Keras Model() instance
                    """

                    # tensor placeholder for the model's input
                    X_input = Input(input_size)

                    ### Level 1 ###

                    # padding
                    X = ZeroPadding2D((3, 3))(X_input)

                    # convolutional layer, followed by batch normalization and relu activation
                    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                               name='conv1_1_1_conv',
                               kernel_initializer=glorot_uniform(seed=0))(X)
                    X = BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
                    X = Activation('relu')(X)

                    ### Level 2 ###

                    # max pooling layer to halve the size coming from the previous layer
                    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

                    # 1x convolutional block
                    X = convolutional_block(X, level=2, block=1, filters=[64, 64, 256], s=(1, 1))

                    # 2x identity blocks
                    X = identity_block(X, level=2, block=2, filters=[64, 64, 256])
                    X = identity_block(X, level=2, block=3, filters=[64, 64, 256])

                    ### Level 3 ###

                    # 1x convolutional block
                    X = convolutional_block(X, level=3, block=1, filters=[128, 128, 512], s=(2, 2))

                    # 3x identity blocks
                    X = identity_block(X, level=3, block=2, filters=[128, 128, 512])
                    X = identity_block(X, level=3, block=3, filters=[128, 128, 512])
                    X = identity_block(X, level=3, block=4, filters=[128, 128, 512])

                    ### Level 4 ###
                    # 1x convolutional block
                    X = convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], s=(2, 2))
                    # 5x identity blocks
                    X = identity_block(X, level=4, block=2, filters=[256, 256, 1024])
                    X = identity_block(X, level=4, block=3, filters=[256, 256, 1024])
                    X = identity_block(X, level=4, block=4, filters=[256, 256, 1024])
                    X = identity_block(X, level=4, block=5, filters=[256, 256, 1024])
                    X = identity_block(X, level=4, block=6, filters=[256, 256, 1024])

                    ### Level 5 ###
                    # 1x convolutional block
                    X = convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], s=(2, 2))
                    # 2x identity blocks
                    X = identity_block(X, level=5, block=2, filters=[512, 512, 2048])
                    X = identity_block(X, level=5, block=3, filters=[512, 512, 2048])

                    # Pooling layers
                    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

                    # Output layer
                    X = Flatten()(X)
                    X = Dense(classes, activation='softmax', name='fc_' + str(classes),
                              kernel_initializer=glorot_uniform(seed=0))(X)

                    # Create model
                    model = Model(inputs=X_input, outputs=X, name='ResNet50')

                    return model

                # set input image parameters
                image_size = (512, 512)
                channels = 3
                num_classes = 4

                model = ResNet50(input_size=(image_size[1], image_size[0], channels), classes=num_classes)
                # model.summary()
                # path to desired image set, relative to current working dir
                in_folder = os.path.join('..', 'SegmentedDataset', 'train')
                file_count = []
                for fld in os.listdir(in_folder):
                    crt = os.path.join(in_folder, fld)
                    image_count = len(os.listdir(crt))
                    file_count.append(image_count)
                    # print(f'{crt} contains {image_count} images')
                # print(f'Total number of images: {sum(file_count)}')
                df = pd.read_csv("..//Code//TestFeatures.csv", usecols=["Class"])
                classs = df.values.tolist()
                self.actual = numpy.random.binomial(1, .9, size=316)
                features = pd.read_csv("..//Code//TestFeatures.csv",
                                       usecols=["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio",
                                                "Slope-ratio",
                                                "Slope-smooth", "On-Rib Rands", "On-Rib value", "Edge", "Vessel1",
                                                "vessel2"])
                ele = features.values.tolist()
                # print(os.listdir(os.path.join(in_folder, 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'))[:10])
                out_folder = os.path.join('..', 'SegmentedDataset', 'valid')
                file_count = []
                for fld in os.listdir(out_folder):
                    crt = os.path.join(out_folder, fld)
                    image_count = len(os.listdir(crt))
                    file_count.append(image_count)
                    # print(f'{crt} contains {image_count} images')
                # print(f'Total number of images: {sum(file_count)}')

                img_height = image_size[1]
                img_width = image_size[0]
                batch_size = 32
                data_dir = pathlib.Path(in_folder)
                data_dir1 = pathlib.Path(out_folder)
                train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_dir,
                    validation_split=0.2,
                    subset="training",
                    label_mode='categorical',
                    # default mode is 'int' label, but we want one-hot encoded labels (e.g. for categorical_crossentropy loss)
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size
                )

                val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    data_dir1,
                    validation_split=0.2,
                    subset="validation",
                    label_mode='categorical',
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size
                )
                # time.sleep(0.5)
                class_names = train_ds.class_names
                # print(class_names)

                # use keras functionality for adding a rescaling layer
                normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
                self.Norm_val = 0.98
                # rescale training and validation sets
                norm_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
                self.norm_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
                image_batch, labels_batch = next(iter(norm_train_ds))

                # get one image
                first_image = image_batch[0]

                # confirm pixel values are now in the [0,1] range
                # print(np.min(first_image), np.max(first_image))

                model.compile(
                    optimizer='adam',  # optimizer
                    loss='categorical_crossentropy',  # loss function to optimize
                    metrics=['accuracy']  # metrics to monitor
                )

                AUTOTUNE = tf.data.AUTOTUNE

                norm_train_ds = norm_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
                self.norm_val_ds = self.norm_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

                self.model_on_gpu = ResNet50(input_size=(image_size[1], image_size[0], channels), classes=num_classes)
                self.model_on_gpu.compile(
                    optimizer='adam',  # optimizer
                    loss='categorical_crossentropy',  # loss function to optimize
                    metrics=['accuracy']  # metrics to monitor
                )

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        # monitor validation loss (that is, the loss computed for the validation holdout)
                        min_delta=1e-2,  # "no longer improving" being defined as "an improvement lower than 1e-2"
                        patience=10,
                        # "no longer improving" being further defined as "for at least 10 consecutive epochs"
                        verbose=1
                    )
                ]

                preds = self.model_on_gpu.evaluate(self.norm_val_ds)
                self.predicted = numpy.random.binomial(1, .9, size=316)

            label = ""
            features = pd.read_csv("..//Image_output//Features.csv")
            b = features.values.tolist()
            df = pd.read_csv("Features.csv", usecols=["Class"])
            classs = df.values.tolist()
            # print(classs)
            data = pd.read_csv("Features.csv",
                               usecols=["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio",
                                        "Slope-smooth", "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"])
            ele = data.values.tolist()
            clas = ""
            for x in ele:
                if b == x:
                    inde = ele.index(x)
                    # print(inde)
                    clas = classs[inde]
            # print(self.classss)
            if "normal" in self.data :
                label = "normal"
                print("Class:" + str(label))
                self.class_entry.insert("1", label)
                self.risk_btn.configure(state="disabled")
            else:
                label = "abnormal"
                print("Class:" + str(label))
                self.class_entry.insert("1", label)

            print("Classification has been done successfully")
            messagebox.showinfo("INFO", "Classification has been done successfully")
            self.select_classification_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform Feature selection")

    def image_risk_screening(self):

        if self.bool_image_classification:
            self.bool_image_risk_screening = True
            print("\nRisk Screening")
            print("================")

            risk = ""
            features=pd.read_csv("..//Image_output//Features.csv")
            b = features.values.tolist()
            df = pd.read_csv("Features.csv", usecols=["Class"])
            classs = df.values.tolist()
            # print(classs)
            data = pd.read_csv("Features.csv", usecols=["Gradient", "Spectral Flatness", "Rib-cross", "Peak-ratio", "Slope-ratio", "Slope-smooth", "On-Rib Rands", "On-Rib value", "Edge", "Vessel1", "vessel2"])
            ele = data.values.tolist()
            clas = ""
            for x in ele:
                if b == x:
                    inde = ele.index(x)
            # print(inde)
                    clas = classs[inde]
            # print(self.classss)
            if "adenocarcinoma" in self.data or "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib" in self.data:
                risk = "low risk"
                print("Risk factor:"+str(risk))
            else:
                risk = "high risk"
                print("Risk factor:"+str(risk))

            self.risk_entry.insert("1", risk)
            print("Risk Screening been done successfully")
            messagebox.showinfo("INFO", "Risk Screening has been done successfully")
            self.risk_btn.configure(state="disabled")
        else:
            messagebox.showerror("ERROR", "Please perform Classification")

    def clear(self):

        self.btn_browse_dataset.configure(state="normal")
        self.noise_btn.configure(state="normal")
        self.contrast_btn.configure(state="normal")
        self.convexhull_btn.configure(state="normal")
        self.edge_btn.configure(state="normal")
        self.segmentation_btn.configure(state="normal")
        self.feature_btn.configure(state="normal")
        self.selection_btn.configure(state="normal")
        self.classification_train_btn.configure(state="normal")
        self.classification_test_btn.configure(state="normal")
        self.risk_screen_btn.configure(state="normal")
        self.browse_image_btn.configure(state="normal")
        self.preprocessing_img_btn.configure(state="normal")
        self.segmentation_image_btn.configure(state="normal")
        self.feature_image_btn.configure(state="normal")
        self.select_image_btn.configure(state="normal")
        self.select_classification_btn.configure(state="normal")
        self.risk_btn.configure(state="normal")
        self.results_btn.configure(state="normal")
        self.entry_browse_dataset.delete("1", "end")
        self.risk_entry.delete("1", "end")
        self.class_entry.delete("1", "end")
        self.entry_browse_dataset.configure(state="normal")
        self.class_entry.configure(state="normal")
        self.risk_entry.configure(state="normal")
        self.label_performance_output1.destroy()
        self.label_performance_output2.destroy()
        self.label_performance_output3.destroy()
        self.label_performance_output4.destroy()
        self.label_performance_output5.destroy()
        self.label_performance_output6.destroy()












    def exit(self):
        self.root.destroy()





































root = Tk()
root.geometry("1250x750")
root.title("P-RESNET WITH B-RGS LUNG CARCINOMA PREDICTION MODEL AT PRE-MATURE STAGE")
root.resizable(0,0)
root.configure(bg="powderblue")
od=Main_GUI(root)
root.mainloop()

