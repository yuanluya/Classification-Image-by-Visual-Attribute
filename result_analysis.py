import re
import numpy as np
import json
import sys
def main():
	#get prediction and label
	pd = sys.argv[1]
	ld = sys.argv[2]
	fp = open(pd, 'r')
	fl = open(ld, 'r')
	fgroundT = open('labels.json', 'r')
	
	#load data calculate average distance with ground truth
	groundT = json.loads(fgroundT.readline())
	predictions = json.loads(fp.readline())
	labels = json.loads(fl.readline())
	
	dist = 0
	for i in labels.keys():
		labels[i] = np.array(eval(labels[i]))
		predictions[i] = np.array(eval(predictions[i]))
		dist += np.linalg.norm(labels[i] - predictions[i])
	print dist / len(labels.keys())
	
	#calculate distance
	distM = np.zeros((len(labels.keys()), len(groundT.keys())))
	for i in range(len(labels.keys())):
		for j in range(len(groundT.keys())):
			if groundT.keys()[j] == "feature_names":
				continue
			distM[i, j] = np.linalg.norm(groundT[groundT.keys()[j]]["features"] - predictions[labels.keys()[i]])
	print "get dist matrix", distM.shape
	distM = distM.argsort()[:, 0: 5]
	
	#calculate accuracy
	category = {}
	correct = 0
	right = False
	for i in range(distM.shape[0]):
		for j in range(distM.shape[1]):
			key = labels.keys()[i][0: 9]
			if groundT.keys()[distM[i, j]] == key:
				correct += 1
				right = True
				break;
		if category.get(key):
			category[key][1] += 1
			category[key][0] += right
		else:
			category[key] = [0 + right, 1]
		right = False
	print correct * 1.0 / len(labels.keys())
	f = open('accuracy_by_category.txt', 'w')
	for name in category.keys():
		f.write(name + ' ' + str(category[name][0]) + " " + str(category[name][1]) + " " + str(1.0 * category[name][0] / category[name][1]) + '\n')
if __name__ == '__main__':
	main()
