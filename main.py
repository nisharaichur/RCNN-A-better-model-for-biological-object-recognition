import tensorflow as tf 
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten
from IPython.core.display import display
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from recurrent_lateral import RNN, RecurrentCell

def gradient_loss(net, data, actualOutput, loss, regularizer):
    with tf.GradientTape() as t:
        t.watch(net.variables)
        predicted = net(data, 4)
        loss_val = tf.reduce_mean([loss(actualOutput, x) for x in predicted.values()])
        norm = normComputation(net.variables)
        loss_regularized = loss_val + (norm * regularizer) 
    gradients = t.gradient(loss_regularized, net.variables)
    return loss_regularized, gradients

def cross_entropy_loss(labels, logits):
    loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss_val)
    
def normComputation(variables):
    norm=0.0
    for x in variables:
            norm = norm + tf.nn.l2_loss(x)
    return norm

def train_rnn(net, data, loss, batch, epochs, lr, train_lbs):
    data_length = data.shape[0]
    loss_list=[]
    for e in range(epochs):
        print("epoch ", e)
        avg_loss = 0.0
        lossval = 0.0
        regularizer = tf.constant(0.0005)
        decayRate = 0.1
        decayStep = 40
        learningRate = lr
        indices = tf.random.shuffle(tf.range(data_length - batch -1))
        v = tuple(tf.zeros(shape = x.shape) for x in net.variables)
        for idx in indices:
            trainData = data[idx:idx+batch,:,:,:]
            #labels = tf.one_hot(train_lbs[idx:idx+batch], 10)
            labels = tf.reshape(train_lbs[idx:idx+batch], shape=[-1])
            labels = tf.cast(labels, tf.int32)
            learningRate = lr * pow(decayRate, (e / decayStep))
            lossval, gradients = gradient_loss(net, trainData, labels, loss, regularizer) 
            v = tuple((x * 0.9) - (learningRate * y) for x, y in zip(v, gradients))
            for x in range(len(net.variables)):
                net.variables[x].assign_add(v[x]) 
            print(" the loss value every time is", lossval)
            avg_loss += lossval / indices.shape[0]
            print("the average_loss = ", avg_loss)
        loss_list.append(avg_loss)
        print("Epoch {} of {} : loss = {}.".format(e+1, epochs, (avg_loss)))
    return loss_list

def main():
	imagLoad = loadmat('test/test.mat')
	net = RNN(imagLoad['images'][0:100,:,:,:].shape, 3)
	loss_list = train_rnn(net,data=tf.convert_to_tensor(imagLoad['images'], dtype="float32"), loss=cross_entropy_loss, batch=100, epochs=20, lr=0.1, train_lbs=imagLoad['targets'])
	plt.plot(loss_list)

if __name__ == '__main__'
	main()
