import tensorflow as tf 
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Flatten
from IPython.core.display import display
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from recurrent_topDown import RNN, RecurrentCell
import json
from datetime import datetime
import pickle 

def gradient_loss(net, data, actualOutput, loss, accuracy, regularizer):
    with tf.GradientTape() as t:
        t.watch(net.variables)
        predicted = net(data, 4)
        loss_val = tf.reduce_mean([loss(actualOutput, x) for x in predicted.values()])
        accu = accuracy(actualOutput, predicted[3])
        loss_regularized = loss_val + (tf.math.reduce_mean([tf.nn.l2_loss(x) for x in net.variables]) * regularizer) 
    gradients = t.gradient(loss_regularized, net.variables)
    return loss_regularized, gradients, accu
    
def cross_entropy_loss(labels, logits):
    loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(loss_val)
 
def accuracy(labels, logits):
    labels = tf.cast(labels, tf.int64)
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

def evaluate(net, data, loss, accuracy, batch, lbs):
    data_length = data.shape[0]
    avg_loss = []
    avg_acc = []
    regularizer = tf.constant(0.0005)
    length = data_length - batch -1
    for idx in range(0, length, batch):
        val_data = data[idx:idx+batch,:,:,:]
        labels = tf.cast(tf.reshape(lbs[idx:idx+batch], shape=[-1]), tf.int32)
        predicted = net(val_data, 4)
        loss_val = tf.reduce_mean([loss(labels, x) for x in predicted.values()])
        loss_regularized = loss_val + (tf.math.reduce_sum([tf.nn.l2_loss(x) for x in net.variables]) * regularizer)
        accval = accuracy(labels, predicted[3])
        #accval = tf.reduce_mean([accuracy(labels, x) for x in predicted.values()])
        avg_loss.append(loss_regularized) 
        avg_acc.append(accval )
    return tf.reduce_mean(avg_loss), tf.reduce_mean(avg_acc)

def train_rnn(net, data, val_data, test_data, loss, accuracy, batch, epochs, lr, train_lbs, val_lbs, test_lbs):
    data_length = data.shape[0]
    regularizer = tf.constant(0.005)
    decayRate = 0.1
    decayStep = 40
    acc_list = []
    error_list = []
    valacc_list = []
    valerror_list = []
    for e in range(epochs):
        avg_loss = []
        avg_acc = []
        length = data_length - batch -1
        v = tuple(tf.zeros(shape = x.shape) for x in net.variables)
        
        if (e % 10 == 0):
            val_loss, val_acc = evaluate(net, val_data, loss, accuracy, batch, val_lbs)
            valacc_list.append(val_acc.numpy())
            valerror_list.append(val_loss.numpy())
            print("Time ", datetime.now().time() )
            print("Epoch {} of {} : Validation loss = {}. Validation Accuracy = {}.".format(e+1, epochs, val_loss, val_acc ))
           
        for idx in range(0, length, batch):
            trainData = data[idx:idx+batch,:,:,:]
            labels = tf.cast(tf.reshape(train_lbs[idx:idx+batch], shape=[-1]), tf.int32)
            learningRate = lr * pow(decayRate, (e / decayStep))
            lossval, gradients, accval = gradient_loss(net, trainData, labels, loss, accuracy, regularizer) 
            v = tuple((x * 0.9) - (learningRate * y) for x, y in zip(v, gradients))
            for i, j in zip(v, net.variables):
                j.assign_add(i)
            avg_loss.append(lossval)
            avg_acc.append(accval)


        acc_list.append(tf.reduce_mean(avg_acc).numpy())
        error_list.append(tf.reduce_mean(avg_loss).numpy())
        print("Time ", datetime.now().time() )
        print("Epoch {} of {} : loss = {}. Accuracy = {}.".format(e+1, epochs, tf.reduce_mean(avg_loss), tf.reduce_mean(avg_acc)))

    print("--------------Testing the model on Test data-----------------------")
    test_loss, test_acc = evaluate(net, test_data, loss, accurac, batch, test_lbs)
    return net.variables, acc_list, error_list, valacc_list, valerror_list, test_loss.numpy(), test_acc.numpy()

def main():
	train_images = loadmat('test/images_with_debris.mat')
	validate_images = loadmat('validate/images_with_debris.mat')
	test_images = loadmat('test_data/images_with_debris.mat')
	net = RNN(train_images['images'][0:100,:,:,:].shape, 3)
	train_data = tf.cast(tf.image.per_image_standardization(train_images['images']), dtype="float32")
	validate_data = tf.cast(tf.image.per_image_standardization(validate_images['images']), dtype="float32")
	test_data = tf.cast(tf.image.per_image_standardization(test_images['images']), dtype="float32")

	net_variables, acc_list, error_list, valacc_list, valerror_list, test_loss, test_acc = train_rnn(net, data=train_data, val_data=validate_data, test_data=test_data, loss=cross_entropy_loss, accuracy=accuracy, batch=100, epochs=100, lr=0.1, train_lbs=train_images['targets'], val_lbs=validate_images['targets'], test_lbs=test_images['targets'])
	metrics = {'train_accuracy': acc_list, 'validation_accuracy': valacc_list, 'train_loss': error_list, 'validation_loss': valerror_list, 'test_loss': test_loss, 'test_acc': test_acc}

	with open("metrics.json", "w") as fp:
	    json.dump(str(metrics), fp)
        
        with open("weights.txt", 'rb') as ffp:
            pickle.dump(net_variables, ffp)

if __name__ == '__main__':
	main()
