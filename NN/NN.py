import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.utils import shuffle

W = np.transpose(pd.read_csv("1078Weather2016.csv").values)
W = np.transpose(W)[:,1:10]
SP = pd.read_csv("2016_1078_Solarpanel.csv")
results = np.array(SP["Generated"])

W, results = shuffle(W,results, random_state=0)

train_size = int(len(results)*(2/3))

# x is input weather data
x_train = W[:train_size,:]
x_test =  W[train_size:,:]

# y is solar panel output data
y_train = results[:train_size]
y_test = results[train_size:]

def nn_model(x_data, input_dim):
    Weights_1 = tf.Variable(tf.random_uniform([input_dim, 16]))
    bias_1 = tf.Variable(tf.zeros([16]))

    layer_1 = tf.add(tf.matmul(x_data, Weights_1), bias_1)
    layer_1 = tf.nn.relu(layer_1)

    Weights_2 = tf.Variable(tf.random_uniform([16, 8]))
    bias_2 = tf.Variable(tf.zeros([8]))

    layer_2 = tf.add(tf.matmul(layer_1, Weights_2), bias_2)
    layer_2 = tf.nn.relu(layer_2)

    Weights_3 = tf.Variable(tf.random_uniform([8, 8]))
    bias_3 = tf.Variable(tf.zeros([8]))

    layer_3 = tf.add(tf.matmul(layer_2, Weights_3), bias_3)
    layer_3 = tf.nn.relu(layer_3)

    Weights_4 = tf.Variable(tf.random_uniform([8, 4]))
    bias_4 = tf.Variable(tf.zeros([4]))

    layer_4 = tf.add(tf.matmul(layer_3, Weights_4), bias_4)
    layer_4 = tf.nn.relu(layer_4)

    Weights_output = tf.Variable(tf.random_uniform([4, 1])) #dtype=tf.float32
    bias_output = tf.Variable(tf.zeros([1]))

    output = tf.add(tf.matmul(layer_4, Weights_output), bias_output)

    return output

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = nn_model(xs,3)

# our mean squared error cost function
cost = tf.reduce_mean(tf.square(output-ys))

# Gradinent Descent optimiztion just discussed above for updating weights and biases
train = tf.train.GradientDescentOptimizer(0.75).minimize(cost)

c_t = []
c_test = []

# run data through neural net
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess, "dataset.csv")

    # run with each sample for cost and train
    for i in range(100):
        for j in range(x_train.shape[0]):
            dicti = {xs:x_train[j:].reshape(-1,3), ys: y_train[j]} #size -1 for unspecified
            sess.run([cost, train], feed_dict = dicti)

        # print each individual cost
        c_t.append(sess.run(cost, feed_dict={xs:x_train.reshape(-1,3), ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:x_test.reshape(-1, 3), ys:y_test}))
        print(i, "%, Cost: ", c_t[i])

    #predict output of test data after training
    predict = sess.run(output, feed_dict={xs:x_test.reshape(-1,3)})

    print("Error: ", predict[-1][0])
    print("OMG HET LEEFT")
