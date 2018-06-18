import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.utils import shuffle

STEP_SIZE = 100
LEARNING_RATE = 0.00005

def make_csv(solar, weather):
    new = weather
    del new["Unnamed: 0"]
    new["number_of_panels"] = solar["Number_of_panels"]
    new["max_power"] = solar["Max_power"]
    new["system_size"] = solar["System_size"]
    new["number_of_inverters"] = solar["Number_of_inverters"]
    new["inverter_size"] = solar["Inverter_size"]
    new["tilt"] = solar["Tilt"]
    return np.array(new)

years = ["2013","2014","2015","2016","2017","2018"]
postal_code = "7559"
W = pd.read_csv("../NN/"+postal_code+ "_" + years[0] + "_W.csv")
SP = pd.read_csv("../NN/"+postal_code+ "_" + years[0] + "_S.csv")
results = np.array(SP["Generated"])
for year in range(1,len(years)):
    W2 = pd.read_csv("../NN/"+postal_code+ "_" + years[year] + "_W.csv")
    W = pd.DataFrame.append(W,W2)
    SP2 = pd.read_csv("../NN/"+postal_code+ "_" + years[year] + "_S.csv")
    SP = pd.DataFrame.append(SP,SP2)
results = np.array(SP["Generated"])
#print(results)
sizes = 1
#print(sizes)

W = (W.values)
train_size = len(results)-365

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
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
    bias_4 = tf.Variable(tf.zeros(4))

    layer_4 = tf.add(tf.matmul(layer_3, Weights_4), bias_4)
    layer_4 = tf.nn.relu(layer_4)

    Weights_output = tf.Variable(tf.random_uniform([4, 1])) #dtype=tf.float32
    bias_output = tf.Variable(tf.zeros([1]))

    model = tf.add(tf.matmul(layer_4, Weights_output), bias_output)

    return model

xs = tf.placeholder("float")
ys = tf.placeholder("float")

model = nn_model(xs,1)

# our mean squared error cost function
cost = tf.reduce_mean(tf.square(model-ys))

# Gradinent Descent optimiztion just discussed above for updating weights and biases
train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
#train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

c_t = []

# run data through neural net
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess, '/home/cait/tweedejaarsproject/NN/dataset.ckpt')

    # run with each sample for cost and train
    for i in range(STEP_SIZE):
       for j in range(x_train.shape[0]):
           dicti = {xs:x_train[j:].reshape(-1,sizes), ys: y_train[j]} #size -1 for unspecified
           sess.run([cost, train], feed_dict = dicti)

        # print each individual cost
       c_t.append(sess.run(cost, feed_dict={xs:x_train.reshape(-1,sizes), ys:y_train}))
       if i%10 == 0:
           print("Step:", i, ", Cost:", c_t[i])

    #predict output of test data after training
    predict = sess.run(model, feed_dict={xs:x_test.reshape(-1,sizes)})
    print(len(predict))
    print(len(x_test.reshape(-1,sizes)))
    AV = sum(np.square(np.mean(predict))-y_test)

    print("Error: ", predict[-1][0])
    #print("OMG HET LEEFT")

    if input('Save model ? [Y/N]').upper() == 'Y':
       saver.save(sess,'/home/cait/tweedejaarsproject/NN/dataset.ckpt')
       print('Model Saved')

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
  # Restore variables from disk.
    # saver.restore(sess, "/home/cait/tweedejaarsproject/NN/dataset.ckpt")
    # print("Model restored.")
  # Check the values of the variables
