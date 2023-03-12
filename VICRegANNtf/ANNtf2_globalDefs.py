"""ANNtf2_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf global definitions

"""

import tensorflow as tf

useLovelyTensors = True

testHarness = False
if(testHarness):
	testHarnessNumSentences = 1	#fixed
	testHarnessNumWords = 3

def printt(text, tensor):
	if(useLovelyTensors):
		printLTstring = text + getLovelyTensorStats(tensor)
		print(printLTstring)
	else:
		print(text, tensor)
	
def printlt(tensor):
	if(useLovelyTensors):
		printLTstring = getLovelyTensorStats(tensor)
		print(printLTstring)
	else:
		print(tensor)
		
def getLovelyTensorStats(tensor):
	printLTstring = "tensor[" + convertTensor1DToString(tensor.shape) + "] n=" + convertTensorElementToString(tf.size(tensor)) + " x\u2208[" + convertTensorElementToString(tf.math.reduce_min(tensor)) + ", " + convertTensorElementToString(tf.math.reduce_max(tensor)) + "] \u03bc=" +  convertTensorElementToString(tf.math.reduce_mean(tensor)) + " \u03c3=" +  convertTensorElementToString(tf.math.reduce_std(tensor))
	return printLTstring

def convertTensorElementToString(t):
	t = tf.squeeze(t).numpy().item()	#t.detach().cpu().numpy().item()
	#t = str(t)
	if(type(t) == float):
		string = '{0:.5f}'.format(t) 
	elif(type(t) == int):
		string = str(t)
	else:
		print("convertTensorElementToString error: type(t) invalid, t = ", t)
		
	return string

def convertTensor1DToString(t):
	#string = '('
	string = ''
	for index in range(tf.size(t)):
		string = string + convertTensorElementToString(t[index])
		if(index != tf.size(t)-1):
			string = string + ','
	#string = string + ')'
	return string
