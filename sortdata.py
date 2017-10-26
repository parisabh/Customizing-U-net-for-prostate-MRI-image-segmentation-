#This file gets the dicom image files and the nrrd segment data files and pad them to make all files 512 by 512 and then
#store it in arrayImage and arrayLabel each of which has size numberOfSlices*512*512 The training slice images where only collected from the copy of train folder.
# The leader file wasn't used for simplicity. This code is written by Parisa Babaheidarian.
from __future__ import division, print_function
import dicom
import os
import numpy as np
import nrrd
import re
from matplotlib import pyplot, cm
import tensorflow as tf
from tempfile import TemporaryFile
import math

PathDicom="C:/Users/Parisa/Documents/DeepMindSegmenatationProject/Copy of train/train/DOI"
Pathnrrd = "C:/Users/Parisa/Documents/DeepMindSegmenatationProject/Copy of train"
#Pathoutput="C:/Users/Parisa/Documents/DeepMindSegmenatationProject/t"
PathDicomTest="C:/Users/Parisa/Documents/DeepMindSegmenatationProject/Copy of test/test/DOI"
PathnrrdTest="C:/Users/Parisa/Documents/DeepMindSegmenatationProject/Copy of test"

lstFilesDCM = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
            
            
len(lstFilesDCM)

lstFilesnrrd = []  # create an empty list, the raw label data files is stored here
for dirName, subdirList, fileList in os.walk(Pathnrrd):
    for filename in fileList:
        if re.match('.*\.nrrd', filename):  # check whether the file's nrrd
            lstFilesnrrd.append(os.path.join(dirName,filename)) 
            
len(lstFilesnrrd) 


#Getting the dimension and allocating space
Rdicom = dicom.read_file(lstFilesDCM[0])   
Rnrrd=nrrd.read(lstFilesnrrd[0])   
ArrayImage = np.zeros((len(lstFilesDCM),512,512),np.float64) #initializing the data arrays, number of slice images is 1555
ArrayLabel = np.zeros((len(lstFilesDCM),512,512),np.int16) 


#Reading image data files and padding and storing
idx=0
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # padding the raw image data
    bb=ds.pixel_array
    bb=np.rot90(bb)
    aa=np.pad(bb, ((math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2)),(math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2))), mode='constant', constant_values=0.0)
    #storing
    ArrayImage[idx, :, :] = aa
    idx=idx+1    
    
idx=0
c=0
NonzeroLabels=[]
NonzeroIds=[]
for idfile in range(len(lstFilesnrrd)):
    # read the file
    nr = nrrd.read(lstFilesnrrd[idfile])
    t=len(nr[0][0][0])
    b=nr[0]
    for slice in range(t):
        # choosing a slice
        bb=b[:,:,slice]
        #padding
        bbb=np.pad(bb, ((math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2)),(math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2))), mode='constant', constant_values=0.0)
        #storing
        ArrayLabel[idx,:, :] = bbb
        idx=idx+1
        if (np.count_nonzero(bbb)>= 1):
            NonzeroIds.append(idx)
            c=c+1
            NonzeroLabels.append(bbb)
            
        

np.unique(ArrayLabel) #it gives the unique class labels      

#saving the training data
np.save('ImagefiletrainCenter',ArrayImage)
np.save('LabelfiletrainCenter',ArrayLabel)  
            
            
            
            
#Getting the test data
lstFilesDCMTest = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(PathDicomTest):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCMTest.append(os.path.join(dirName,filename))
            
            
len(lstFilesDCMTest)

lstFilesnrrdTest = []  # create an empty list, the raw label data files is stored here
for dirName, subdirList, fileList in os.walk(PathnrrdTest):
    for filename in fileList:
        if re.match('.*\.nrrd', filename):  # check whether the file's nrrd
            lstFilesnrrdTest.append(os.path.join(dirName,filename)) 
            
len(lstFilesnrrdTest) 

ArrayImageTest = np.zeros((len(lstFilesDCMTest),512,512),np.float64) #initializing the data arrays, number of slice images is 1555
ArrayLabelTest = np.zeros((len(lstFilesDCMTest),512,512),np.int16) 

idx=0
for filenameDCMTest in lstFilesDCMTest:
    # read the file
    ds = dicom.read_file(filenameDCMTest)
    bb=ds.pixel_array
    # padding the raw image data
    aa=np.pad(ds.pixel_array, ((math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2)),(math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2))), mode='constant', constant_values=0.0)
    #storing
    ArrayImageTest[idx, :, :] = aa
    idx=idx+1    
    
idx=0
for idfile in range(len(lstFilesnrrdTest)):
    # read the file
    nr = nrrd.read(lstFilesnrrdTest[idfile])
    t=len(nr[0][0][0])
    b=nr[0]
    for slice in range(t):
        # choosing a slice
        bb=b[:,:,slice]
        #padding
        bbb=np.pad(bb, ((math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2)),(math.floor((512-len(bb))/2),512-len(bb)-math.floor((512-len(bb))/2))), mode='constant', constant_values=0.0)
        #storing
        ArrayLabelTest[idx,:, :] = bbb
        idx=idx+1
        
        
#saving the test files
np.save('ImagefiletestCenter1.npy',ArrayImageTest)
np.save('LabelfiletestCenter1.npy',ArrayLabelTest)  
                   
           



   