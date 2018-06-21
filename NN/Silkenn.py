"""Simple tutorial for using TensorFlow to compute polynomial regression.
Parag K. Mital, Jan. 2016"""
# %% Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# %% Let's create some toy data
plt.ion()
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
ax.scatter(xs, ys)
fig.show()
plt.draw()
print(xs, ys)

years = ["2013","2014","2015","2016","2017","2018"]
postal_code = "7559"
W = pd.read_csv("../data/"+postal_code+ "_" + years[0] + "_W.csv")
SP = pd.read_csv("../data/"+postal_code+ "_" + years[0] + "_S.csv")
results = np.array(SP["Generated"])
for year in range(1,len(years)):
    W2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_W.csv")
    W = pd.DataFrame.append(W,W2)
    SP2 = pd.read_csv("../data/"+postal_code+ "_" + years[year] + "_S.csv")
    SP = pd.DataFrame.append(SP,SP2)
results = np.array(SP["Generated"])
nr_features = 13

W = (W.values)

train_size = len(results)-365

train_X = W[:train_size,:].transpose()
test_X =  W[train_size:,:].transpose()
train_Y = results[:train_size].transpose()
test_Y = results[train_size:].transpose()

print(len(W), len(W[0]))

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 13 # 1st layer number of features
n_hidden_2 = 13 # 2nd layer number of features
n_input = 1990 # days

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, 1])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1]))
}

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_input/batch_size)
        # Loop over all batches
        count = 0
        for i in range(total_batch):
            batch_x = train_X[count:(batch_size*i)]
            batch_y = train_Y[count:(batch_size*i)]
            count = (batch_size*i)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

# # %% We create a session to use the graph
# n_epochs = 1000
# with tf.Session() as sess:
#     # Here we tell tensorflow that we want to initialize all
#     # the variables in the graph so we can use them
#     sess.run(tf.global_variables_initializer())
#
#     # Fit all training data
#     prev_training_cost = 0.0
#     for epoch_i in range(n_epochs):
#         for (x, y) in zip(train_X, train_Y):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
#
#         training_cost = sess.run(
#             cost, feed_dict={X: train_X, Y: train_Y})
#         print(training_cost)
#
#         if epoch_i % 100 == 0:
#             ax.plot(xs, Y_pred.eval(
#                 feed_dict={X: train_X}, session=sess),
#                     'k', alpha=epoch_i / n_epochs)
#             fig.show()
#             plt.draw()
#
#         # Allow the training to quit if we've reached a minimum
#         if np.abs(prev_training_cost - training_cost) < 0.000001:
#             break
#         prev_training_cost = training_cost
# ax.set_ylim([-3, 3])
# fig.show()
# plt.waitforbuttonpress()
