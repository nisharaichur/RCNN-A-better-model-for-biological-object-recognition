import tensorflow as tf 
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten
#ImageOcclusionDataset will create a dataset and returns a labels 
#import ImageOcclusionDataset as occlusion 
from RecurrentNets import RNN, RecurrentCell


#Divides the data into batches
def train_rnn(net, data, loss, batch, epochs, lr,train_lbs):
    data_length = data.shape[0]
    for e in range(epochs):
        print("epoch ", e)
        avg_loss = 0.0
        lossval = 0.0
        regularizer = tf.constant(0.0005)
        decayRate = 0.1
        decayStep = 40
        indices = tf.random.shuffle(tf.range(data_length - batch -1))
        v = tuple(tf.zeros(shape = x.shape) for x in net.variables)
        for idx in indices:
            trainData = data[idx:idx+batch,:,:,:]
            labels = tf.one_hot(train_lbs[idx:idx+batch],10)
            lossval, v = gradient_loss(net, trainData, labels,loss,lr,regularizer,v,decayRate,decayStep,epochs)
            avg_loss += lossval / indices.shape[0]
        print("Epoch {} of {} : loss = {}.".format(e, epochs, avg_loss), end='/r')


#Calculates the gradient and updates the model variables
def gradient_loss(net, data, actualOutput, loss, lr, regularizer, v, decayRate, decayStep, epochs):
    with tf.GradientTape() as t:
        predicted = net(data, 4)
        norm = 0.0
        loss_val = tf.reduce_mean([tf.reduce_mean(loss(actualOutput,x)) for x in predicted.values()])
    for x in net.variables:
        norm = norm + tf.linalg.norm(x)
    loss_regularized = loss_val + (norm * regularizer) 
    gradients = t.gradient(loss_regularized, net.variables)
    learningRate = lr * pow(decayRate, (epochs / decayStep))
    v = tuple((x * 0.9) - (learningRate * y) for x, y in zip(v, gradients))
    for x in range(len(net.variables)):
        net.variables[x].assign_add(v[x]) 
    return loss_regularized, v


def main():
	(train_imgs, train_lbs), (test_imgs, test_lbs) = tf.keras.datasets.mnist.load_data()
	train_imgs = train_imgs.reshape(-1, 28, 28, 1)
	oneImage = train_imgs[0:100, :, :, :]
	oneImage = tf.convert_to_tensor(oneImage,dtype="float32")
	net = RNN(oneImage.shape, 3)
	train_rnn(net,data=tf.convert_to_tensor(train_imgs[0:500,:,:,:],dtype="float32"),loss=tf.nn.softmax_cross_entropy_with_logits,batch=100,epochs=10,lr=0.1,train_lbs=train_lbs)



if __name__=='__main__':
    main()
