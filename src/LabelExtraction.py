import numpy as np
import sys
from seg_ftr_extraction import *
import ast

# sess = sys.argv[1]
# gender = sys.argv[2]
# improNo = sys.argv[3]

# inputFileName = "Ses0" + sess + gender + "_impro0" + improNo + ".txt"
# outputFileName = sess + gender + improNo 

# scriptNo = sys.argv[4]
# inputFileName = "Ses0" + sess + gender + "_script0" + improNo + "_" + scriptNo +  ".txt"
# outputFileName = sess + gender + improNo + scriptNo + ".txt"

inputFileName = "./utteranceLabels/mergedUtterancesRefined.txt"

emotionMap = {
	"exc" : "[1,0,0,0,0]",
	"fru" : "[0,1,0,0,0]",
	"hap" : "[0,0,1,0,0]",
	"neu" : "[0,0,0,1,0]",
	"sad" : "[0,0,0,0,1]"
}

#Ses01F_impro01_F000
# Ses01F_script01_1_F000
# total useful utterances : 6277
def storeUtteranceEmotions(inputFilename,outputFilename):
	with open(inputFilename) as f:
		j = 1
		for line in f:
			if (line[0]=='['):
				fileName = line.split()[3] 
				emotion = line.split()[4]
				# if emotion in emotionMap:
				out = open(outputFilename + str(j) + ".txt",'w')
				segments = get_segment_features_from_file("/media/rg/Important Files/IEMOCAP_full_release/Session" + fileName[4] + "/sentences/wav/" + fileName[:-5] + "/" + fileName + ".wav")
				emotionFeature = emotionMap[emotion] 
				
				for i in segments:
					out.write(str([i.tolist(),ast.literal_eval(emotionFeature)]))
					out.write('\n')

				out.close()
				print("utterance " + str(j) + " " + fileName + " written")
				j = j+1

storeUtteranceEmotions(inputFileName,"segmentLabels/")
