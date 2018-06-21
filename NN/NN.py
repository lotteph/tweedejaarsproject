import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.utils import shuffle

# Disables the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

STEP_SIZE = 500
LEARNING_RATE = 0.0001

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
postal_code = ["2134", "2201", "7325", "7559"]

SP = False
W = False
for code in range(0,len(postal_code)):
    for year in range(0,len(years)):
        W2 = pd.read_csv("./data/"+postal_code[code]+ "_" + years[year] + "_W.csv")
        if type(W) != type(False):
            W = pd.DataFrame.append(W,W2)
        else:
            W = W2
        SP2 = pd.read_csv("./data/"+postal_code[code]+ "_" + years[year] + "_S.csv")
        if type(SP) != type(False):
            SP = pd.DataFrame.append(SP,SP2)
        else:
            SP = SP2
results = np.array(SP["Generated"])
W["time"] = 0
W = make_csv(SP, W)

years = ["2013","2014","2015","2016","2017","2018"]
postal_code = "7559"

W = pd.read_csv("../NN/data/"+postal_code+ "_" + years[0] + "_W.csv")
SP = pd.read_csv("../NN/data/"+postal_code+ "_" + years[0] + "_S.csv")
results = np.array(SP["Generated"])
for year in range(1,len(years)):
    W2 = pd.read_csv("../NN/data/"+postal_code+ "_" + years[year] + "_W.csv")
    W = pd.DataFrame.append(W,W2)
    SP2 = pd.read_csv("../NN/data/"+postal_code+ "_" + years[year] + "_S.csv")
    SP = pd.DataFrame.append(SP,SP2)
results = np.array(SP["Generated"])
W = (W.values)

train_size = len(results)-365

sizes = W.shape[1]

x_train = W[:train_size,:]#.transpose()
x_test =  W[train_size:,:]#.transpose()
y_train = results[:train_size]#.transpose()
y_test = results[train_size:]#.transpose()

c_t = []

# run data through neural net
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saved_model = input('Name of the model: ')
    #saver.restore(sess, '/models/' + saved_model + '.ckpt')

    # run with each sample for cost and train
    for i in range(STEP_SIZE):
        for j in range(x_train.shape[0]):
            dicti = {xs:x_train[j:].reshape(-1, sizes), ys: y_train[j]} #size -1 for unspecified
            sess.run([cost, train], feed_dict = dicti)

        # print each individual cost
        c_t.append(sess.run(cost, feed_dict={xs:x_train.reshape(-1,sizes), ys:y_train}))
        if i%10 == 0:
            print("Step:", i, ", Cost:", c_t[i])

    #predict output of test data after training
    predict = sess.run(model, feed_dict={xs:x_test.reshape(-1,sizes)})

    print("Error: ", predict[-1][0])

    plt.plot(predict)
    plt.show()

    if input('Save model ? [Y/N] ').upper() == 'Y':
        new_model_name = input('Please name the file: ')
        saver.save(sess, './models/' + new_model_name + '.ckpt')
        print('Model Saved')

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
  # Restore variables from disk.
    # saver.restore(sess, "/home/cait/tweedejaarsproject/NN/dataset.ckpt")
    # print("Model restored.")
  # Check the values of the variables
