#!/usr/bin/env python
# coding: utf-8

# # Neural Networks with MNIST

# ## Data Ingestion

# In[545]:


# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# import base packages 
import numpy as np
import pandas as pd
import matplotlib


# In[546]:


#import relevant machine learning packages
import tensorflow as tf
import keras 


# In[547]:


#other packages
import time


# In[548]:


#set working directory
import os
os.chdir('C:\\Users\\R\\Desktop\\MSDS 422\\Assignment 6')


# In[549]:


#import dataset from Tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# ### Note:
# Models 1, 2 and 3 will analyze how increasing the number of nodes (10, 20, 50) impacts a 2 layer model

# ## Model 1 (2 layers,  10 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)

# In[550]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 10
n_hidden2 = 10
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[551]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden2, n_outputs, name = "outputs")


# In[552]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[553]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[554]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[555]:


init = tf.global_variables_initializer() 


# In[556]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels})\n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val)  ')


# In[557]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val))  


# ## Model 2 (2 layers, 20 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)

# In[690]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 20
n_hidden2 = 20
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[691]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden2, n_outputs, name = "outputs")


# In[692]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[693]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[694]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[695]:


init = tf.global_variables_initializer() 


# In[696]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val)\n            ')


# In[697]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val))      


# ## Model 3 (2 layers, 50 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)

# In[646]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 20
n_hidden2 = 20
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[647]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden2, n_outputs, name = "outputs")


# In[648]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[649]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[650]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[652]:


init = tf.global_variables_initializer() 


# In[653]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val)')


# In[654]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val))   


# ### Note:
# It is apparent that Model 2 (20 nodes for each layer) results in the best performance (both time and accuracy) of the three models. This same analysis will be applied to 5 layer models (Models 4, 5 and 6).

# ## Model 4 (5 layers, 10 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)

# In[566]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 10
n_hidden2 = 10
n_hidden3 = 10
n_hidden4 = 10
n_hidden5 = 10
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[567]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name = "hidden3", activation = tf.nn.relu) 
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name = "hidden4", activation = tf.nn.relu) 
    hidden5 = tf.layers.dense(hidden4, n_hidden5, name = "hidden5", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden5, n_outputs, name = "outputs")


# In[568]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[569]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[570]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[571]:


init = tf.global_variables_initializer() 


# In[572]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val) ')


# In[615]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val))  


# ## Model 5 (5 layers, 20 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)

# In[658]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 20
n_hidden2 = 20
n_hidden3 = 20
n_hidden4 = 20
n_hidden5 = 20
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[659]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name = "hidden3", activation = tf.nn.relu) 
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name = "hidden4", activation = tf.nn.relu) 
    hidden5 = tf.layers.dense(hidden4, n_hidden5, name = "hidden5", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden5, n_outputs, name = "outputs")


# In[660]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[661]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[662]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[663]:


init = tf.global_variables_initializer() 


# In[664]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val)')


# In[665]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val))  


# ## Model 6 (5 hidden layers, 50 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)

# In[667]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 50
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_hidden5 = 50
n_outputs = num_classes

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[668]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name = "hidden3", activation = tf.nn.relu) 
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name = "hidden4", activation = tf.nn.relu) 
    hidden5 = tf.layers.dense(hidden4, n_hidden5, name = "hidden5", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden5, n_outputs, name = "outputs")


# In[669]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[670]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[671]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[672]:


init = tf.global_variables_initializer() 


# In[673]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val)')


# In[674]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val)) 


# ### Note:
# It appears that adding more layers does not necessarily improve accuracy or time for this data set. In fact, the 5 layer 10 node model was the worst performer (avg training accuracy: 82%, avg validation accuracy: 75.6%) compared to all 2 layer and 5 layer models. In both 2 and 5 layer models, increasing the number of nodes generally increased performance. The best performing model, however, was Model 2 (2 layer 20 node) which averaged 100% training accuracy and 95.6% validation accuracy. 
# 
# In the next model, I will build off of Model 2 and adjust other hyperparameters including number of epochs, batch size and learning rate.

# ## Model 7 (2 layers, 20 nodes, 20 epochs, batch size = 200, learning rate = .01)

# In[624]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.01
n_epochs = 20
batch_size = 200

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 10
n_hidden2 = 10
n_outputs = num_classes

#Create placeholder variables
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[625]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden2, n_outputs, name = "outputs")


# In[626]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[627]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[628]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[629]:


init = tf.global_variables_initializer() 


# In[630]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val) ')


# In[631]:


#Calculate averages
print("Average Training Accuracy:", np.mean(acc_train))
print("Average Training Loss:", np.mean(loss_train))
print("Average Validation Accuracy:", np.mean(acc_val))
print("Average Validation Loss:", np.mean(loss_val))  


# ### Note:
# It seems that increasing the number of epochs and batch size while decreasing the learning rate does not increase model performance. 10 epochs, batch size = 100 and learning rate = .5 seem to be suitable parameters for the model. The one parameter that has shown a positive impact on model performance is adjusting the number of nodes per layer. 
# 
# In the next model (Model 8) I will use these parameters and look at how significantly increasing and then staggering the nodes per layer impacts model performance. The first layer will have 3x more nodes than the second layer (300 and 100 nodes).
# 

# ## Model 8 (2 layers, 1 layer = 300 nodes, 2 layer = 100 nodes, epoch=10, batch size = 100, learning rate = .5)

# In[698]:


tf.reset_default_graph()

# Set hyperparameters
learning_rate = 0.5
n_epochs = 10
batch_size = 100

image_size = 784
num_classes = 10

n_inputs = image_size
n_hidden1 = 300
n_hidden2 = 100
n_outputs = num_classes

#Create placeholder variables
X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.float32, shape = (None), name = 'Y')


# In[699]:


#Specify classifying method
with tf.name_scope("dnn"): 
    hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1", activation = tf.nn.relu) 
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name = "hidden2", activation = tf.nn.relu) 
    logits = tf.layers.dense(hidden2, n_outputs, name = "outputs")


# In[700]:


#Specify method to compute loss
with tf.name_scope("loss"): 
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits) 
    loss = tf.reduce_mean(xentropy, name = "loss")


# In[701]:


#Specify training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
    training_op = optimizer.minimize(loss)


# In[702]:


#Specify evaluation method
labels = tf.argmax(y, -1)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[703]:


init = tf.global_variables_initializer() 


# In[704]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    init.run()\n    for epoch in range(n_epochs): \n        for iteration in range( mnist.train.num_examples // batch_size): \n            X_batch, y_batch = mnist.train.next_batch(batch_size) \n            sess.run(training_op, feed_dict ={ X: X_batch, y : y_batch}) \n            acc_train = accuracy.eval(feed_dict ={ X: X_batch, y : y_batch}) \n            acc_val = accuracy.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            loss_train = loss.eval(feed_dict = { X: X_batch, y : y_batch})\n            loss_val = loss.eval(feed_dict ={ X: mnist.validation.images, y: mnist.validation.labels}) \n            print("Epoch:", epoch, "---Training accuracy:---", acc_train, "Training Loss:",loss_train,"---Validation accuracy:---", acc_val, "Validation Loss:", loss_val)')


# In[705]:


#Calculate averages
training_average = "Average Training Accuracy:", np.mean(acc_train)
print(training_average)
training_loss_average ="Average Training Loss:", np.mean(loss_train)
print(training_loss_average)
validation_average ="Average Validation Accuracy:", np.mean(acc_val)
print(validation_average)
validation_loss_average = "Average Validation Loss:", np.mean(loss_val)
print(training_loss_average)


# ## Conclustion:
# 
# Of the seven models tested, Model 2 was the best overall performer in terms of accuracy, loss minimization and computation time. Model 2 had the following hyperparameters: 2 hidden layers, 20 nodes per layer, learning rate = .5, epochs = 10 and batch size = 100. Model 2 achieved an average of 100% training accuracy and 95.6% validation accuracy in 1 min 48 seconds. Model 8  also resulted in similar prediction results. Model 8 had the same parameters as Model 2 but had 300 nodes in the first layer and 100 in the second layer. It achieved an average training accuracy score of 100% and an average validation score of 98%. The main difference is Model 7 took 7 min 30 seconds of computation time compared to 1 min 48 seconds. Therefore, Model 2 was deemed the optimal model.
# 
# Model 4 was by far the worst performer of the models with average training accuracy of 82% and 75.6% average validation accuracy in 1 min 48 seconds. Model 4 had the following hyperparameters: 5 hidden layers, 10 nodes per layer, learning rate = .5, epochs = 10, batch size = 100)
# 
# A conclusion from these results is that in general, increasing the number of nodes per layer has a positive impact on accuracy. However, as nodes per layer increase so does computation time. Another conclusion is that increasing the number of layers does not necessarily lead to more accurate results. This may be a result of the data not being very complex, causing the algorithm to overfit the training data and not generalize to new data.
# 
# Using Model 2 as a reference, increasing other hyperparameters like batch size and number of epochs while adjusting learning rate had mostly negative effects on model performance.

# In[ ]:




