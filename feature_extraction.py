import xml.etree.ElementTree as et
from sets import Set
import numpy as np
import json

def getVector(fileName):
	vector = Set([])
	vectorColor = Set([])
	tree = et.parse(fileName)
	root = tree.getroot()
	for subcategory in root:
		for concept in subcategory:
			anatomy = concept.find('anatomy')
			if anatomy != None:
				temp = anatomy.text.replace('\n', ' ')
				temp = temp.replace(' ', ' ')
				temp = temp.split()
				for attribute in temp:
					if attribute[-3: ] == '$$$':
						attribute = attribute[0: -3]
					vector.add(attribute)
			color = concept.find('colour_patterns')
			if color != None:
				temp = color.text.replace('\n', ' ')
				temp = temp.replace(' ', ' ')
				temp = temp.split()
				for attribute in temp:
					vector.add(attribute)
	return list(vector)#, list(vectorColor)

def getMatrix(totalFeatures, fileName):
	matrix = []
	names = []
	ID = []
	length = len(totalFeatures)
	tree = et.parse(fileName)
	root = tree.getroot()
	cc = 0
	text_file = open("ids.txt", "w")
	for subcategory in root:
		for concept in subcategory:
			#print concept.attrib['synsetID']
			names.append(concept.attrib['name'])
			ID.append(concept.attrib['synsetID'])
			text_file.writelines(concept.attrib['synsetID']+'\n')
			anatomy = concept.find('anatomy')
			#print anatomy
			featureVector = np.zeros(length)
			if anatomy != None:
				temp = anatomy.text.replace('\n', ' ')
				temp = temp.replace(' ', ' ')
				temp = temp.split()
				for attribute in temp:
					if attribute[-3: ] == '$$$':
						attribute = attribute[0: -3]
						featureVector[totalFeatures.index(attribute)] = 0
					else:
						featureVector[totalFeatures.index(attribute)] = 1
			color = concept.find('colour_patterns')
			if color != None:
				temp = color.text.replace('\n', ' ')
				temp = temp.replace(' ', ' ')
				temp = temp.split()
				for attribute in temp:
					featureVector[totalFeatures.index(attribute)] = 1
			matrix.append(featureVector)
	text_file.close()
	return np.array(matrix), names,ID

def main():
	vector = getVector('ANIMALS_structured_final.xml')
	print vector
	Matrix78, names,ID = getMatrix(vector, 'ANIMALS_structured_final.xml')
	print len(names), len(ID)
	std = np.sum(Matrix78, axis = 0)
	print np.shape(Matrix78), std
	for i in range(std.shape[0]):
		if std[i] <= 5:
			print vector[i], std[i],
			for j in range(len(names)):
				if Matrix78[j, i] == 1:
					print names[j], 
			print '\n',
	#for i in range(std.shape[0]):
		#if std[i]  == 0.5:
			#print vector[i]
	dictionary = {}
	for name in range(len(names)):
		dictionary[ID[name]] = {'name': names[name], 'features': list(Matrix78[name, :])}
	dictionary['feature_names'] = vector
	labels = json.dumps(dictionary)
	f = open('labels.json', 'w')
	f.write(labels)
	dist = 0
	mini = 80;
	maxi = 0;
	num = 0
	for i in range(np.shape(Matrix78)[0]):
		for j in range(i + 1, np.shape(Matrix78)[0]):
			temp = np.linalg.norm(Matrix78[i] - Matrix78[j])
			if temp == 0:
				num += 1
				#print names[i], names[j]
			if temp < mini:
				mini = temp
			if temp > maxi:
				maxi = temp
			dist += temp
	dist /= np.shape(Matrix78)[0] * (np.shape(Matrix78)[0] - 1) / 2
	#print dist, mini, maxi, num



if __name__ == '__main__':
	main()