import numpy as np
import sys
from seg_ftr_extraction import *
import ast

sess = sys.argv[1]
gender = sys.argv[2]
improNo = sys.argv[3]

inputFileName = "Ses0" + sess + gender + "_impro0" + improNo + ".txt"
outputFileName = sess + gender + improNo 

# scriptNo = sys.argv[4]
# inputFileName = "Ses0" + sess + gender + "_script0" + improNo + "_" + scriptNo +  ".txt"
# outputFileName = sess + gender + improNo + scriptNo + ".txt"

emotionMap = {
	"exc" : "[1,0,0,0,0]",
	"fru" : "[0,1,0,0,0]",
	"hap" : "[0,0,1,0,0]",
	"neu" : "[0,0,0,1,0]",
	"sad" : "[0,0,0,0,1]"
}

def storeUtteranceEmotions(inputFilename,outputFilename):
	with open(inputFilename) as f:
		j = 0
		for line in f:
			if (line[0]=='['):
				fileName = line.split()[3] 
				emotion = line.split()[4]
				if emotion in emotionMap:
					out = open(outputFilename + fileName[-4:] + '.txt','w')
					segments = get_segment_features_from_file("/media/rg/Important Files/IEMOCAP_full_release/Session" + sess + "/sentences/wav/" + fileName[:-5] + "/" + fileName + ".wav", 16000)
					emotionFeature = emotionMap[emotion] 
					
					for i in segments:
						out.write(str([i.tolist(),ast.literal_eval(emotionFeature)]))
						out.write('\n')

					out.close()
				print("utterance " + str(j) + " written")
				j = j+1

storeUtteranceEmotions("/media/rg/Important Files/IEMOCAP_full_release/Session"+ sess + "/dialog/EmoEvaluation/" + inputFileName,"segmentLabels/" + outputFileName)
