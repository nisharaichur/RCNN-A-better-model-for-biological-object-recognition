import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten
from IPython.core.display import display
import matplotlib.pyplot as plt
from scipy.io import loadmat


class RecurrentCell(tf.Module):
    def __init__(self,filterSize, inChannel, outChannel, activation, fc=False):
        self.filterSize = filterSize
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.activation = activation
        self.fc = fc 
        if self.fc == True:
            self.kernelW = tf.Variable(tf.random.normal(shape=(self.outChannel, self.inChannel)))
            self.b = tf.Variable(tf.zeros(shape=(1, self.outChannel)), dtype="float32")
        else:
            self.kernelW = tf.Variable(tf.random.normal(shape=(self.filterSize, self.filterSize, self.inChannel, self.outChannel), name="FwdKernel"))
            self.kernelU = tf.Variable(tf.random.normal(shape=(self.filterSize, self.filterSize, self.outChannel, self.outChannel), name="lateralKernel"))
            self.kernelT = tf.Variable(tf.random.normal(shape=(self.filterSize, self.filterSize, self.outChannel, self.outChannel), name="topdownKernel"))
            self.b = tf.Variable(tf.zeros(shape=(self.outChannel, )), dtype="float32", name="bias")
        
    def __call__(self, inputImage,dialatedImage=None, topDown=False):   
        if self.fc == True:
            inputImage = Flatten()(inputImage)
            fwd = tf.tensordot(inputImage, tf.reshape(self.kernelW, shape=(inputImage.shape[1], -1)), axes=1) 
            h = fwd + self.b
        else: 
            #if dialatedImage != None:   
            fwd = tf.nn.conv2d(inputImage, self.kernelW, padding="SAME", strides=1)
            #rec = tf.nn.conv2d(lateralImage, self.kernelU, padding="SAME", strides=1)
            h = fwd + self.b
            if topDown == True:
                #deconvImage = tf.nn.conv2d_transpose(dialatedImage, self.kernelU, output_shape=h.shape, padding="SAME", strides=2 )
                deconvImage = tf.nn.conv2d_transpose(dialatedImage, self.kernelT, output_shape=h.shape, padding="SAME", strides=2 )
                h = h + deconvImage          
            h = self.activation(h)
        return(h)
            

class RNN(tf.Module):
    def __init__(self,imageShape,hiddenUnit): 
        filterSize = 3
        self.pooling = [2, 16, None]
        outChannel = 32
        self.hiddenUnit = hiddenUnit
        self.layer = []
        activation = tf.nn.relu
        inChannel = imageShape[-1]
        for i in range(self.hiddenUnit):
            if i == (self.hiddenUnit-1):
                self.layer.append(RecurrentCell(filterSize, inChannel, outChannel=10, activation=tf.nn.sigmoid, fc=True))
            else:
                self.layer.append(RecurrentCell(filterSize, inChannel, outChannel, activation))
            inChannel = outChannel
        
    def __call__(self,inputImage,timeSteps):
        states = []
        dictOutputs = {}
        states.append(None)
        states.append(tf.zeros(shape=(100, 16, 16, 32)))
        states.append(None)
        for i in range(timeSteps):
            newStates = []
            x = inputImage
            for enu in range(len(states)):
                if enu ==0:
                    x = self.layer[enu](x, states[enu+1],topDown=True)
                
                if enu==1:
                    x = self.layer[enu](x)
                    
                if self.pooling[enu] == None:
                    x = self.layer[enu](x)
                    newStates.append(None)
                    #dictOutputs[i] = tf.nn.softmax(x)
                    dictOutputs[i] = x
                    continue  
                newStates.append(x)
                x = tf.nn.local_response_normalization(x, depth_radius=5, bias=1, alpha=0.001, beta=0.5)  
                x = tf.nn.max_pool(x, ksize=self.pooling[enu], strides=2, padding="VALID") 
            states = newStates 
        return(dictOutputs)
        
        
