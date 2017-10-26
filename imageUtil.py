
'''
This code fetches data and perform pre and post processing as well as data_augmentation. 
author: Parisa Babaheidarian (The original source code was written by jakeret and Parisa added major changes to the available code.
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
#import glob
from numpy.random import normal
import numpy as np
from PIL import Image
import math
#import tflearn #for data augmentation
#from tflearn.data_augmentation import ImageAugmentation

#img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_rotation(max_angle=25.)

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    """
    
    channels = 1
    n_class = 3 #0,1,2
   # n_class=2
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()   
        train_data = self._process_data(data)
        #labels = self._process_labels(label)
        
        train_data, label = self._post_process(train_data, label)
        labels = self._process_labels(label)
        
        nx = data.shape[1]
        ny = data.shape[0]
        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):
        if (self.n_class==3):
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.int16)
            l=label.reshape(1,ny,nx)
            labels[:,:, 0] = np.minimum(1+np.array(label),np.maximum(1-np.array(label),np.zeros((1,ny, nx), dtype=np.int16)))
            labels[:,:, 1] =np.minimum(2-np.array(label),label)
            labels[:,:, 2] =np.maximum(np.array(label)-1,np.zeros((1,ny, nx), dtype=np.int16), dtype=np.int16)
        if (self.n_class==2):
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            
        return labels
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, label):
        """
        Post processing hook that can be used for data augmentation
        
        We add i.i.d. Gaussian noise followed by a random rotation
        """
      
        #Adding i.i.d. Gaussian noise element-wise with sigma=0.1
        noise=0.1*data #20 dB
        data=data+normal(0,noise)
        #Rotating the image by a random rotation alpha
        alpha=np.random.randint(0,25) # a random rotation with maximum 25 degrees rotation
        #print("alpha is",alpha)
        alphar=math.radians(alpha)
        datap=np.zeros((512,512),dtype=np.float64)
        labelp=np.zeros((512,512),dtype=np.int16)
        #Rotating
        for i in range(512):
            for j in range(512):
                if(i==0):
                    if(math.sqrt(i*i+j*j)*math.cos(math.pi/2+alphar)>=0 and math.sqrt(i*i+j*j)*math.cos(math.pi/2+alphar)<=511 and math.sqrt(i*i+j*j)*math.sin(math.pi/2+alphar)>=0 and math.sqrt(i*i+j*j)*math.sin(math.pi/2+alphar)<=511):
                        datap[int(round(math.sqrt(i*i+j*j)*math.cos(math.pi/2+alphar))),int(round(math.sqrt(i*i+j*j)*math.sin(math.pi/2+alphar)))]=data[i,j]
                        labelp[int(round(math.sqrt(i*i+j*j)*math.cos(math.pi/2+alphar))),int(round(math.sqrt(i*i+j*j)*math.sin(math.pi/2+alphar)))]=label[i,j]
                        
                    
                else:
                    if(math.sqrt(i*i+j*j)*math.cos(math.atan(j/i)+alphar)>=0 and math.sqrt(i*i+j*j)*math.cos(math.atan(j/i)+alphar)<=511 and math.sqrt(i*i+j*j)*math.sin(math.atan(j/i)+alphar)>=0 and math.sqrt(i*i+j*j)*math.sin(math.atan(j/i)+alphar)<=511):
                        datap[int(round(math.sqrt(i*i+j*j)*math.cos(math.atan(j/i)+alphar))),int(round(math.sqrt(i*i+j*j)*math.sin(math.atan(j/i)+alphar)))]=data[i,j]
                        labelp[int(round(math.sqrt(i*i+j*j)*math.cos(math.atan(j/i)+alphar))),int(round(math.sqrt(i*i+j*j)*math.sin(math.atan(j/i)+alphar)))]=label[i,j]
                
                
            
        #Interpolating        
        for i in range(1,511): #ignoring boundaries
            for j in range(1,511): #four-way neighborhood interpolation
                a=np.array([labelp[i,j+1],labelp[i,j-1],labelp[i+1,j],labelp[i-1,j]])
                if(labelp[i,j]==0 and np.argmax(np.bincount(a)!=0)):
                    labelp[i,j]=np.argmax(np.bincount(a)) #replace with the most frequent label in the neighborhood
                if(datap[i,j]!=datap[i,j+1] and datap[i,j]!=datap[i,j-1] and datap[i,j]!=datap[i+1,j] and datap[i,j]!=datap[i-1,j]):
                    datap[i,j]=(datap[i,j+1]+datap[i,j-1]+datap[i+1,j]+datap[i-1,j])/4 #replace with mean value of the neighbourhood
                
        return datap, labelp
        return data, label
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.
    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 3):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        print("idx is",idx)       
        return np.rot90(self.data[idx]), self.label[idx] #The images saved in training need to be rotated by 90 degrees to match the label assignment
