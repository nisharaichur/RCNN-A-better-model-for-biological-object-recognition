import tensorflow as tf 
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten
#ImageOcclusionDataset will create a dataset and returns a labels 
#import ImageOcclusionDataset as occlusion 
from RecurrentNets import RNN, RecurrentCell


#Divides the data into batches
def train_rnn(net, data, batch, loss, epochs=10, lr=0.1):
    data_length = data.shape[0]
    for e in range(epochs):
        avg_loss = 0.0
        lossval = 0.0
        index = 0
        decayRate = 0.1
        decayStep = 40
        batches_per_epoch = data_length/batch
        v = tuple(tf.zeros(shape = x.shape) for x in net.variables)
        while(index != data_length):
            lossval, gradients = rnn_step(net, data, index, batch, loss, lr)
            avg_loss += lossval
            index = index + batch
            learningRate = lr * pow(decayRate, (epochs / decayStep))
            v = tuple(x * 0.9 - (learningRate * y) for x, y in zip(v, gradients))
            for x in range(len(net.variables)):
                net.variables[x].assign_add(v[x])  
            #print(avg_loss)
        print("Epoch {} of {}; loss={}.".format(e, epochs, (avg_loss/batches_per_epoch), end="\r"))   
'''
def cross_loss(predict, a, variables, regularizer):
    error = 0
    norm = 0
    for keys in predict:
        error = error + tf.reduce_sum(a * tf.math.log(tf.clip_by_value(predict[keys], 1e-10, 1.0)) + (1-a) * tf.math.log(tf.clip_by_value((1 - predict[keys]), 1e-10, 1.0)))
    
    for x in variables:
        norm = norm + tf.linalg.norm(x)
    return(-error * norm) 
'''

#Iterates over mini-batch of 100
def rnn_step(net, data, index, batch, loss, lr):
    avg_loss = 0
    loss_val = 0
    regularizer = tf.constant(0.0005)
    meanGradient = tuple(tf.zeros(shape=x.shape) for x in net.variables)
    meanLoss=0
    for i in range(batch): 
        gradients, loss_val=gradient_loss(net, data[i+index : i+1+index, :, :], train_lbs[i], loss, lr, regularizer)
        meanGradient = tuple(tf.math.add(x, y) for x, y in zip(meanGradient, gradients))
        meanLoss = tf.math.add(loss_val, meanLoss)
    meanGradient = tuple(x/batch for x in meanGradient)  
    meanLoss = meanLoss/batch
    return meanLoss, meanGradient

#Calculates the gradient and updates the model variables
def gradient_loss(net, data, actualOutput, loss, lr, regularizer):
    with tf.GradientTape() as t:
        norm = 0
        t.watch(net.variables)
        prediction = net(data, 4)
        loss_val = tf.reduce_mean([tf.reduce_mean(loss(x, tf.one_hot(actualOutput, 10))) for x in prediction.values()])
        for x in net.variables:
            norm = norm + tf.linalg.norm(x)
        loss_regularized = loss_val + (norm * regularizer) 
    gradients = t.gradient(loss_regularized, net.variables)
    #print(gradients)
    return gradients, loss_val

def main():
	#Intially trained a model on mnist dataset 
	(train_imgs,train_lbs),(test_imgs,test_lbs)=tf.keras.datasets.mnist.load_data()
	train_imgs=train_imgs.reshape(-1, 28, 28, 1)
	net=RNN(train_imgs[0:1, :, :].shape, 3)
	train_rnn(net,data=tf.convert_to_tensor(train_imgs[0:50,:,:,:], dtype="float32"), batch=10, loss=tf.nn.softmax_cross_entropy_with_logits)	


if __name__=='__main__':
    main()
