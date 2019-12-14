import tensorflow as tf 
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten
#ImageOcclusionDataset will create a dataset and returns a labels 
#import ImageOcclusionDataset as occlusion 
from RecurrentNets import RNN, RecurrentCell


#Divides the data into batches
def train_rnn(net,data,batch,loss,train_lbs,epochs=10,lr=0.001):
    data_length=data.shape[0]
    for e in range(epochs):
        avg_loss = 0.0
        lossval = 0.0
        index=0
        while(index!=data_length):
            lossval=rnn_step(net,data,index,batch,loss,lr,train_lbs=train_lbs)
            avg_loss+=lossval/data_length
            index=index+batch
        print("Epoch {} of {}; loss={}.".format(e,epochs,avg_loss),end="\r") 


def cross_loss(predict, a, variables, regularizer):
    error = 0
    norm = 0
    for keys in predict:
        error = error + tf.reduce_sum(a * tf.math.log(tf.clip_by_value(predict[keys], 1e-10, 1.0)) + (1-a) * tf.math.log(tf.clip_by_value((1 - predict[keys]), 1e-10, 1.0)))
    
    for x in variables:
        norm = norm + tf.linalg.norm(x)
    return(-error * norm) 

#Iterates over mini-batch of 100
def rnn_step(net,data,index,batch,loss,lr,train_lbs):
    avg_loss=0
    loss_val=0
    for i in range(batch):
        loss_val=gradient_loss(net,data[i+index:i+1+index,:,:],train_lbs[i],loss,lr)
        avg_loss+=loss_val  
    return avg_loss

#Calculates the gradient and updates the model variables
def gradient_loss(net, data, actualOutput, loss, lr, regularizer):
    with tf.GradientTape() as t:
        t.watch(net.variables)
        prediction=net(data,4)
        loss_val=loss(prediction, tf.one_hot(actualOutput, 10), net.variables, regularizer)
    gradients = t.gradient(loss_val, net.variables)
    print(len(net.variables))
    for g, v in zip(gradients, net.variables):
        v.assign_sub(lr * g)
    return(loss_val)


def main():
	#Intially trained a model on mnist dataset 
	(train_imgs,train_lbs),(test_imgs,test_lbs)=tf.keras.datasets.mnist.load_data()
	train_imgs=train_imgs.reshape(-1, 28, 28, 1)
	net=RNN(train_imgs[0:1,:,:].shape,3)
	train_rnn(net,data=tf.convert_to_tensor(train_imgs[0:50,:,:,:],dtype="float32"),batch=10,loss=cross_loss)


if __name__=='__main__':
    main()
