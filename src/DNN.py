import tensorflow as tf
import numpy as np
import ast
import shutil
import zipfile
import os
from utterFtrExtraction import getUtteranceFeature

maxEpochs = 2
batchSize = 10000
inputNodes = 480
outputNodes = 5
layer = [inputNodes, 50, 50, 50, outputNodes]

x = tf.placeholder(tf.float32, [None, inputNodes])
y = tf.placeholder(tf.float32, [None, outputNodes])

def neuralNetworkModel(data):
	hiddenLayer1 = {'weights' : tf.Variable(tf.random_normal([layer[0], layer[1]])),
				'biases' : tf.Variable(tf.random_normal([layer[1]]))}

	hiddenLayer2 = {'weights' : tf.Variable(tf.random_normal([layer[1], layer[2]])),
				'biases' : tf.Variable(tf.random_normal([layer[2]]))}

	hiddenLayer3 = {'weights' : tf.Variable(tf.random_normal([layer[2], layer[3]])),
				'biases' : tf.Variable(tf.random_normal([layer[3]]))}

	outputLayer = {'weights' : tf.Variable(tf.random_normal([layer[3], layer[4]])),
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
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
	# cost = tf.reduce_mean(tf.squared_difference(y, prediction))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(maxEpochs):
			epoch_loss = 0
			f = open("segmentLabels/merged.txt")
			lines = f.readlines()
			f.close()
			for k in range((len(lines)/batchSize)):
				batchLines = lines[k:k+batchSize]
				epoch_x = []
				epoch_y = []
				for line in batchLines:
					epoch_x.append(np.array(ast.literal_eval(line)[0]))
					epoch_y.append(np.array(ast.literal_eval(line)[1]))
				
				result = sess.run([optimizer,cost,prediction], feed_dict = {x : epoch_x, 
		 													 y : epoch_y})
				epoch_loss+=result[1]
				# print(result[2])
			print("Epoch" + str(epoch) + " completed with loss " + str(epoch_loss))
		storeModel(sess,"storedParameters")

def restoreModel(session,filename):
	saver = tf.train.Saver()
	unzip = zipfile.ZipFile("./"+filename+".zip")
	unzip.extractall("./temp")
	unzip.close()
	saver.restore(session,"./temp/"+filename)
	shutil.rmtree("./temp")
	return session

def storeModel(session,filename):
	saver = tf.train.Saver()
	saver.save(session,"./temp/"+filename)
	shutil.make_archive(filename, 'zip', "./temp")
	shutil.rmtree("./temp")

def sortFunc(inputString):
	return int(inputString[:-4])

def predict(data, inputPath, outputFileName, threshold = 5, storeLabels=True):
	prediction = neuralNetworkModel(data)
	with tf.Session() as sess:
		sess = restoreModel(sess,"storedParameters")
		out = open(outputFileName,'wa')
		files = sorted(os.listdir(inputPath),key = sortFunc)
		for fileName in files:
			if (os.stat(inputPath + fileName).st_size != 0):
				f = open(inputPath+fileName)
				lines = f.readlines()
				f.close()
				epoch_x = []
				epoch_y = []
				for line in lines:
					epoch_x.append(np.array(ast.literal_eval(line)[0]))
					epoch_y.append(ast.literal_eval(line)[1])

				result = sess.run(prediction,feed_dict = {x : epoch_x})
				utteranceFeatures = getUtteranceFeature(result,threshold)
		
				if (not storeLabels):
					epoch_y = ['NA']
				label = epoch_y[0]

				out.write(str([utteranceFeatures,label]))
				out.write('\n')
				print(fileName)		
		out.close()

# trainNetwork(x)
predict(x,"segmentLabels/","utteranceFeatures/utteranceFeatures.txt",storeLabels=True)