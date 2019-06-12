import tensorflow as tf
import numpy as np
import csv
#for weight initialization

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape = shape,stddev=0.05))

#for bias initialization

def new_biases(length):
    return tf.Variable(tf.constant(shape=[length],value=0.05))

#Fully Connected layer

#returns : layer and weight of this layer

def fully_connected_layer(input,num_inputs,num_outputs,ifbiase,activate):
    weights = new_weights([num_inputs,num_outputs])
    if ifbiase:
        biases = new_biases(num_outputs)
        logits = tf.matmul(input,weights) + biases
    else:
        logits = tf.matmul(input,weights)
    if activate==-1:
        return logits,weights
    if activate==0:
        logits = tf.nn.sigmoid(logits)
    elif activate ==1 :
        logits = tf.nn.tanh(logits)
    else:
        logits = tf.nn.relu(logits)
    return logits,weights


#CSVReader files unload the data by following the instruction given in config file

import CSVReader as param

#required values for forward propagation in neural network

nodes = param.nodes
num_features = param.num_features
num_classes = param.num_classes
num_layers = param.num_layers
bias_include = param.bias_include
learning_rate= param.learning_rate
epochs = param.epochs
mini_batches = param.mini_batches
activation = param.activation
train_x = param.train_x
train_y = param.train_y
input_size = param.input_size
ptype = param.ptype


# NN Construction


# place holder variables
x = tf.placeholder(tf.float32,[None,num_features])
y_true = tf.placeholder(tf.float32,[None,num_classes])

#list which will store the weights of each layer
weights = []
# x_norm = normalized(x)

#Neural Network
#following neural network has settings in accordance with regression type problem
#if neural network needed to use classification type problem, you need to change function
#of cross-entropy and y_pred

if len(nodes) == 2:
    layer,extra = fully_connected_layer(x,nodes[0],nodes[1],bias_include,-1)
    weights.append(extra)
else:
    layer,extra = fully_connected_layer(x,nodes[0],nodes[1],bias_include,activation)
    weights.append(extra)
    for i in range(2,num_layers-1):
        layer,extra = fully_connected_layer(layer,nodes[i-1],nodes[i],bias_include,activation)
        weights.append(extra)

    layer,extra = fully_connected_layer(layer,nodes[-2],nodes[-1],bias_include,-1)
    weights.append(extra)

if not ptype :
	y_pred = tf.nn.relu(layer)
	cross_entropy = tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
else:
	y_pred = tf.nn.softmax(layer)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=layer)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#creates the next batch of size given: num

def next_batch(num):
    idx = np.arange(input_size)
    np.random.shuffle(idx)
    idx = idx[:num]
    random_data = np.array([ train_x[i] for i in idx])
    random_labels = np.array([ train_y[i] for i in idx])
    return random_data,random_labels


#starting the session

session = tf.Session()
session.run(tf.global_variables_initializer())

whole_feed_dict = {x:train_x,y_true:train_y}

#oprtimization optimizes the variables weights and biases

def optimization(num_iterations,epochs,verbose=True):

    #verbose is to print the cost output after every epoch

    for j in range(1,epochs+1):
        for i in range(num_iterations):
            x_batch, y_true_batch = next_batch(mini_batches)
            feed_dict = {x: x_batch,y_true: y_true_batch}
            session.run(optimizer,feed_dict=feed_dict)
            if verbose and i==num_iterations-1:
                print(j,'th epoch cost: ',session.run(cost,feed_dict=feed_dict),sep='')

    if verbose:
        print("\n\nPrediction: ")
        print(session.run([y_pred],feed_dict=whole_feed_dict))

num_iterations = int((input_size)/mini_batches)
optimization(num_iterations,epochs)


import os

path=''

def csv_weights_write():
    global path
    path = os.path.join(os.getcwd(),'Weights')
    i=1
    while True:
        if not os.path.exists(path):
            os.mkdir(path)
            break
        if i>=2:
            path=path[:(-1-len(str(i)))]
        path+='_'+str(i)
        i+=1
    print("Weights are stored in "+ path)
    w = session.run(weights,feed_dict=whole_feed_dict)
    for i in range(len(w)):
        filename = os.path.join(path,'w'+str(i+1)+'.csv')
        with open(filename,'w') as fl:
            csvwriter = csv.writer(fl)
            csvwriter.writerows(w[i])

csv_weights_write()
param.print_equation(os.path.join(path,'equation.txt'))

