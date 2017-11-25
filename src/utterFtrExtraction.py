import numpy as np
import ast
import os

def getUtteranceFeature(inputFileName,threshold):
	# each column of segmentFeatures is a single segment distribution
	f = open(inputFileName)
	lines = f.readlines()
	segmentFeatures = []
	labels = []
	for i in lines:
		segmentFeatures.append(ast.literal_eval(i)[0])
		labels.append(ast.literal_eval(i)[1])
	
	segmentFeatures = np.array(segmentFeatures)
	segmentFeatures = np.transpose(np.array(segmentFeatures))
	_,noOfSegments = segmentFeatures.shape
	minEmotion = [min(i) for i in segmentFeatures]
	maxEmotion = [max(i) for i in segmentFeatures]
	meanEmotion = [(sum(i)/noOfSegments) for i in segmentFeatures]
	thresholdAppliedVector = [[1 if j>threshold else 0 for j in i] for i in segmentFeatures]
	thresholdEmotion = [sum(i)/noOfSegments for i in thresholdAppliedVector]

	utteranceFeatures = minEmotion + maxEmotion + meanEmotion + thresholdEmotion
	return utteranceFeatures,labels[0]


# print(getUtteranceFeature("output.txt",5))
path = './segmentProbabilities/'

def writeUtteranceFeatures(outputFileName, path, threshold):
	out = open(outputFileName,'w')
	for filename in os.listdir(path):
		utteranceFeatures,label = getUtteranceFeature(path+filename,threshold)
		out.write(str([utteranceFeatures,label]))
		out.write('\n')
	out.close()

writeUtteranceFeatures("bro.txt",path,5)


