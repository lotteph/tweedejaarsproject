import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from tensorflow.contrib import learn
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
import scipy.ndimage

# Disables the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 20
BATCH_SIZE = 32
DISPLAY_STEP = 1

# Network Parameters
N_HIDDEN_1 = 32
N_HIDDEN_2 = 200
N_HIDDEN_3 = 200
N_CLASSES = 1

# Create a model with 3 layers that have RELU activation and an output layer
# with linear activation
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Launch the session
def run_session(x_train, x_test, y_train, y_test, offset):
    n_input = x_train.shape[1]
    total_len = x_train.shape[0]

    # Placeholders for the neural network
    x = tf.placeholder("float", [None, 18])
    y = tf.placeholder("float", [None])

    # Store weight & bias for the layers
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, N_HIDDEN_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([N_HIDDEN_3, N_CLASSES], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([N_HIDDEN_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([N_HIDDEN_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([N_HIDDEN_3], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([N_CLASSES], 0, 0.1))
    }

    # Construct model and transpose it so the prediction and y have the same shape
    pred = multilayer_perceptron(x, weights, biases)
    pred = tf.transpose(pred)

    # Define the loss function and the optimizer
    cost = tf.reduce_mean(tf.square(pred-y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Launch the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        cost_array = []

        # Train the neural network
        for epoch in range(TRAINING_EPOCHS):
            avg_cost = 0.
            total_batch = int(total_len/BATCH_SIZE)
            # Loop over all batches
            for i in range(total_batch-1):
                batch_x = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                batch_y = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                # Run optimization for the backwards pass and cost to get loss value
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                                y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            cost_array.append(avg_cost)

            # Display logs per epoch step
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch+1), "of", TRAINING_EPOCHS)
                print("cost=", "{:.9f}".format(avg_cost))
                print("[*]----------------------------")
                for i in range(10):
                    print ("label value:", batch_y[i], \
                        "estimated value:", p[0][i])
                print("[*]============================")

        print("Optimization Finished!")

        # Test the neural network and compute the error
        predicted_vals = sess.run(pred, feed_dict={x: x_test})
        return predicted_vals

        # Save the model
        # if input('Save model ? [Y/N] ').upper() == 'Y':
        #     if not os.path.exists('./models'):
        #         os.makedirs('./models')
        #     new_model_name = input('Please name the file: ')
        #     saver.save(sess, './models/' + new_model_name + '.ckpt')
        #     print('Model Saved')
