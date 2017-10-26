#This is the main source call the trainer to train the U_Net. This code is written by Parisa Babaheidarian.

from __future__ import division, print_function
import numpy as np
import tensorflow as tf
import imageUtil
from imageUtil import BaseDataProvider 
from imageUtil import SimpleDataProvider
import tf_unet
from tf_unet import unet
from matplotlib import pyplot, cm
import math

#loading train data and test data, only train data is used for training
ArrayImage=np.load('ImagefiletrainCenter.npy','r+')
ArrayLabel=np.load('LabelfiletrainCenter.npy','r+')  
ArrayImageTest=np.load('ImagefiletestCenter1.npy','r+')
ArrayLabelTest=np.load('LabelfiletestCenter1.npy','r+')  


data_provider = SimpleDataProvider(ArrayImage, ArrayLabel)

#Unet network architecture 


net = unet.Unet(channels=data_provider.channels, 
                n_class=data_provider.n_class, 
                layers=3, 
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                )
                
     
#training the network
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(data_provider, "./unet_trained", 
                     training_iters=20, 
                     epochs=7, 
                     dropout=0.5, 
                     display_step=1)    
                     
                     
# testing the prediction result on test data
data_provider = SimpleDataProvider(ArrayImageTest, ArrayLabelTest)

for i in range (100):
    x_test, y_test = data_provider(1)
    prediction = net.predict("./unet_trained/model.cpkt", x_test)  
    labels=np.round(prediction) #rounding probabilities to 0 and 1
    TP=np.zeros((100,3),dtype=np.int16) #TP for 3 classes
    TP[i,0]=(512*512-np.count_nonzero(labels[0,:,:,0]-y_test[0,:,:,0]))/(512*512)
    TP[i,1]=(512*512-np.count_nonzero(labels[0,:,:,1]-y_test[0,:,:,1]))/(512*512)
    TP[i,2]=(512*512-np.count_nonzero(labels[0,:,:,2]-y_test[0,:,:,2]))/(512*512)
    
print("Average True positive of background class is: ", np.mean(TP[:,0], axis=0) 
print("Average True positive of class label 1 is: ", np.mean(TP[:,1], axis=0)  
print("Average True positive of class label 2 is: ", np.mean(TP[:,2], axis=0)    
  
   
    
    

#plotting the results
pyplot.imshow(x_test[0,:,:])
pyplot.set_cmap('jet')
pyplot.show()  
pyplot.imshow(labels[0,:,:,1]+2*labels[0,:,:,2]) #superimposing labels from each class output
pyplot.set_cmap('jet')
pyplot.show()   
"""                   
