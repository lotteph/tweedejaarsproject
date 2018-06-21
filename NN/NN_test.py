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

# Disables the warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 500
BATCH_SIZE = 32
DISPLAY_STEP = 1

# Network Parameters
N_HIDDEN_1 = 32
N_HIDDEN_2 = 200
N_HIDDEN_3 = 200
N_HIDDEN_4 = 256
N_CLASSES = 1

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

data = dict()
data["7325"] = range(2013, 2019)
data["7559"] = range(2013, 2019)
data["2134"] = range(2009, 2019)
data["2201"] = range(2013, 2019)
data["6591"] = range(2015, 2019)
data["3481"] = range(2015, 2019)
data["5384"] = range(2013, 2016)

# Train on multiple postal codes
postal_codes = ["7559", "2134", "2201", "6591", "3481", "5384", "7325"]

# Train on a single postal code
# postal_codes = ["2134"]

SP = False
W = False
print(data.keys())
for code in postal_codes:
    for year in data[code]:
        W2 = pd.read_csv("./data/"+str(code)+ "_" + str(year) + "_W.csv")
        if type(W) != type(False):
            W = pd.DataFrame.append(W, W2)
        else:
            W = W2
        SP2 = pd.read_csv("./data/"+ str(code) + "_" + str(year) + "_S.csv")
        if type(SP) != type(False):
            SP = pd.DataFrame.append(SP, SP2)
        else:
            SP = SP2
results = np.array(SP["Generated"]/SP["Number_of_panels"]/SP["Max_power"])
W["time"] = 0
W = make_csv(SP, W)

train_size = len(results)-365*6

X_train = W[:train_size,:]
X_test =  W[train_size:,:]
Y_train = results[:train_size]
Y_test = results[train_size:]

offset = SP["Number_of_panels"].values[train_size:]*SP["Max_power"].values[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# Y_train = scaler.fit_transform(Y_train)
# Y_test = scaler.fit_transform(Y_test)

n_input = X_train.shape[1]
total_len = X_train.shape[0]

# tf Graph input
x = tf.placeholder("float", [None, 18])
y = tf.placeholder("float", [None])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # # Hidden layer with RELU activation
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, N_HIDDEN_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([N_HIDDEN_2, N_HIDDEN_3], 0, 0.1)),
    # 'h4': tf.Variable(tf.random_normal([N_HIDDEN_3, N_HIDDEN_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([N_HIDDEN_3, N_CLASSES], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_HIDDEN_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([N_HIDDEN_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([N_HIDDEN_3], 0, 0.1)),
    # 'b4': tf.Variable(tf.random_normal([N_HIDDEN_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([N_CLASSES], 0, 0.1))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
pred = tf.transpose(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cost_array = []


    # Training cycle
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0.
        total_batch = int(total_len/BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_y = Y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        cost_array.append(avg_cost)

        # sample prediction
        label_value = batch_y
        estimate = p

        # Display logs per epoch step
        if epoch % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
            print("[*]----------------------------")
            for i in range(10):
                print ("label value:", label_value[i], \
                    "estimated value:", estimate[0][i])
            print("[*]============================")

    print("Optimization Finished!")

    predicted_vals = sess.run(pred, feed_dict={x: X_test})
    # error = np.square(predicted_vals-Y_test)*offset
    error = np.sqrt(sum(np.square((predicted_vals-Y_test)*offset))/len(Y_test))
    print("error:")
    print(np.mean(error))
    # print(np.sqrt(sum(sum(error))/len(Y_test)))

    # plt.plot(cost_array)
    # plt.ylim(0, 0.0005)
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Cost")
    # plt.show()

    predicted_vals = scipy.ndimage.gaussian_filter(predicted_vals, 5)
    Y_test = scipy.ndimage.gaussian_filter(Y_test, 5)
    plt.plot(predicted_vals[0], label='predicted output', color="red")
    plt.plot(Y_test, label='real output', color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("Neural network predicted vs real output")
    plt.show()

    if input('Save model ? [Y/N] ').upper() == 'Y':
        if not os.path.exists('./models'):
            os.makedirs('./models')
        new_model_name = input('Please name the file: ')
        saver.save(sess, './models/' + new_model_name + '.ckpt')
        print('Model Saved')
