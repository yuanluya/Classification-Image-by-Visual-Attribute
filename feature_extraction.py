import xml.etree.ElementTree as et
from sets import Set
import numpy as np

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
	length = len(totalFeatures)
	tree = et.parse(fileName)
	root = tree.getroot()
	for subcategory in root:
		for concept in subcategory:
			names.append(concept.attrib['name'])
			anatomy = concept.find('anatomy')
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
	return np.array(matrix), names

def main():
	vector = getVector('visa_dataset/ANIMALS_structured_final.xml')
	print vector
	Matrix78, names = getMatrix(vector, 'visa_dataset/ANIMALS_structured_final.xml')
	print np.shape(Matrix78), np.std(Matrix78, axis = 0)
	dist = 0
	mini = 80;
	maxi = 0;
	num = 0
	for i in range(np.shape(Matrix78)[0]):
		for j in range(i + 1, np.shape(Matrix78)[0]):
			temp = np.linalg.norm(Matrix78[i] - Matrix78[j])
			if temp == 0:
				num += 1
				print names[i], names[j]
			if temp < mini:
				mini = temp
			if temp > maxi:
				maxi = temp
			dist += temp
	
	dist /= np.shape(Matrix78)[0] * (np.shape(Matrix78)[0] - 1) / 2
	print dist, mini, maxi, num



if __name__ == '__main__':
	main()