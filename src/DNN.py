import tensorflow as tf
import numpy as np

maxEpochs = 10
batchSize = 100
inputNodes = 750
outputNodes = 5
layer = [inputNodes, 256, 256, 256, outputNodes]

x = tf.placeholder(tf.float32, [None, inputNodes])
y = tf.placeholder(tf.float32, [None, outputNodes])
inputData = ////////////////////////////////////////////////////////
inputLabels = ///////////////////////////////////////////////////////
noOfTrainingSamples = len(inputData)


def neuralNetworkModel(data):
	hiddenLayer1 = {'weights' : tf.Variable(tf.random_normal([layer[0], layer[1]])),
				'biases' : tf.Variable(tf.random_normal([layer[1]]))}

	hiddenLayer2 = {'weights' : tf.Variable(tf.random_normal([layer[1], layer[2]])),
				'biases' : tf.Variable(tf.random_normal([layer[2]]))}

	hiddenLayer3 = {'weights' : tf.Variable(tf.random_normal([layer[3], layer[2]])),
				'biases' : tf.Variable(tf.random_normal([layer[3]]))}

	outputLayer = {'weights' : tf.Variable(tf.random_normal([layer[4], layer[3]])),
				'biases' : tf.Variable(tf.random_normal([layer[4]]))}


	net1 = tf.add(tf.matmul(data, hiddenLayer1['weights']),hiddenLayer1['biases'])
	outputHiddenLayer1 = tf.nn.relu(net1)

	net2 = tf.add(tf.matmul(outputHiddenLayer1, hiddenLayer2['weights']),hiddenLayer2['biases'])
	outputHiddenLayer2 = tf.nn.relu(net2)

	net3 = tf.add(tf.matmul(outputHiddenLayer2, hiddenLayer3['weights']),hiddenLayer3['biases'])
	outputHiddenLayer3 = tf.nn.relu(net3)

	output = tf.add(tf.matmul(outputHiddenLayer3, outputLayer['weights']),outputLayer['biases'])

	return output

def trainNetwork(data):
	prediction = neuralNetworkModel(data)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
	gradientDescentOptimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
	optimizer = gradientDescentOptimizer.minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(maxEpochs):
			for k in range((noOfTrainingSamples/batchSize)):
				epoch_x = inputData[k:k+batchSize]
				epoch_y = inputLabels[k:k+batchSize]
				result = sess.run([cost,optimizer], feed_dict = {x : epoch_x, 
																 y : epoch_y})


trainNetwork(x)