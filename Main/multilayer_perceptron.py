import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from tensorflow.contrib import learn
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
import scipy.ndimage
import re

# Disable the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_mlp():
    # Parameters
    LEARNING_RATE = 0.001
    TRAINING_EPOCHS = 1500
    BATCH_SIZE = 32
    DISPLAY_STEP = 1

    # Network Parameters
    N_HIDDEN_1 = 32
    N_HIDDEN_2 = 200
    N_HIDDEN_3 = 200
    N_CLASSES = 1

    #make this into pandas
    def make_pandas(solar, weather):
        new = weather
        del new["Unnamed: 0"]
        new["number_of_panels"] = solar["Number_of_panels"]
        new["max_power"] = solar["Max_power"]
        new["system_size"] = solar["System_size"]
        new["number_of_inverters"] = solar["Number_of_inverters"]
        new["inverter_size"] = solar["Inverter_size"]
        new["tilt"] = solar["Tilt"]
        return new

    def create_weather_pandas(input_set):
        SP = False
        W = False

        data, postal_codes = create_postal_codes(input_set)

        for code in postal_codes:
            for year in data[code]:
                W2 = pd.read_csv("../"+input_set+"/"+str(code)+ "_" + str(year) + "_W.csv")
                if type(W) != type(False):
                    W = pd.DataFrame.append(W,W2)
                else:
                    W = W2
                SP2 = pd.read_csv("../"+input_set+"/"+ str(code) + "_" + str(year) + "_S.csv")
                if type(SP) != type(False):
                    SP = pd.DataFrame.append(SP,SP2)
                else:
                    SP = SP2
        results = np.array(SP["Generated"]/SP["Number_of_panels"]/SP["Max_power"])
        W["time"] = 0
        W = make_pandas(SP,W)
        offset = SP["Number_of_panels"].values[-1]*SP["Max_power"].values[-1]
        return results, W, offset

    def create_postal_codes(input_set):
        postal_code = []
        d = dict()
        files = os.listdir("../"+input_set+"/")
        for name in files:
            codes = re.search("(\d{4})\_\d{4}", name)
            if codes != None:
                codes = codes.group(1)
                years = re.search("(\d{4})\_[A-Z]", name).group(1)
                if codes not in postal_code:
                    postal_code.append(codes)
                    d[codes] = [years]
                elif years not in d[codes]:
                    d[codes].append(years)
        return d, postal_code

    def set_array_size(postal_codes):
        nr_codes = len(postal_codes)
        if (nr_codes > 1):
            set_size = 1
        else:
            set_size = 0
        return set_size

    # Create x and y data
    y_train, x_train, _ = create_weather_pandas("Train")
    y_test, x_test, offset = create_weather_pandas("Test")

    # Normalize the x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    n_input = x_train.shape[1]
    total_len = x_train.shape[0]

    # Placeholders for the neural network
    x = tf.placeholder("float", [None, 18])
    y = tf.placeholder("float", [None])

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
    return predicted_vals[0]*offset

    # error = np.sqrt(sum(np.square((predicted_vals[0]-y_test)*offset)))/len(y_test)
    # print("error:", error)
    #
    #
    # # Save the model
    # if input('Save model ? [Y/N] ').upper() == 'Y':
    #     if not os.path.exists('./models'):
    #         os.makedirs('./models')
    #     new_model_name = input('Please name the file: ')
    #     saver.save(sess, './models/' + new_model_name + '.ckpt')
    #     print('Model Saved')
