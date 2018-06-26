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
TRAINING_EPOCHS = 20
BATCH_SIZE = 32
DISPLAY_STEP = 1

# Network Parameters
N_HIDDEN_1 = 32
N_HIDDEN_2 = 200
N_HIDDEN_3 = 200
N_HIDDEN_4 = 256
N_CLASSES = 1
N = 9

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
data["2134"] = range(2009, 2018)
data["2201"] = range(2013, 2019)
data["6591"] = range(2015, 2019)
data["3481"] = range(2015, 2019)
data["5384"] = range(2013, 2016)

# Train on multiple postal codes
postal_codes = ["7559", "2201", "6591", "3481", "5384", "7325", "2134"]

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
train_size = len(results)-365

x_train = W[:train_size,:]
x_test =  W[train_size:,:]
y_train = results[:train_size]
y_test = results[train_size:]

offset = SP["Number_of_panels"].values[train_size:]*SP["Max_power"].values[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.fit_transform(y_test)

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
    error = np.sqrt(sum(np.square((predicted_vals[0]-y_test)*offset)))/len(y_test)
    print("error:", error)

    # Plot the test data and the predicted values
    predicted_vals = scipy.ndimage.gaussian_filter(predicted_vals*offset, 5)
    y_test = scipy.ndimage.gaussian_filter(y_test*offset, 5)
    plt.plot(predicted_vals[0], label='predicted output', color="red")
    plt.plot(y_test, label='real output', color="blue")
    plt.legend()
    plt.xlabel("time (days)")
    plt.ylabel("solar panel output (kWh)")
    plt.title("Neural network predicted vs real output")
    plt.show()

    # Save the model
    if input('Save model ? [Y/N] ').upper() == 'Y':
        if not os.path.exists('./models'):
            os.makedirs('./models')
        new_model_name = input('Please name the file: ')
        saver.save(sess, './models/' + new_model_name + '.ckpt')
        print('Model Saved')
