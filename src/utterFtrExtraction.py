import numpy as np
import ast
import os

def getUtteranceFeature(segmentFeatures,threshold):
	segmentFeatures = np.array(segmentFeatures)
	segmentFeatures = np.transpose(segmentFeatures)
	_,noOfSegments = segmentFeatures.shape
	minEmotion = [min(i) for i in segmentFeatures]
	maxEmotion = [max(i) for i in segmentFeatures]
	meanEmotion = [(sum(i)/noOfSegments) for i in segmentFeatures]
	thresholdAppliedVector = [[1 if j>threshold else 0 for j in i] for i in segmentFeatures]
	thresholdEmotion = [sum(i)/noOfSegments for i in thresholdAppliedVector]

	utteranceFeatures = minEmotion + maxEmotion + meanEmotion + thresholdEmotion
	return utteranceFeatures


