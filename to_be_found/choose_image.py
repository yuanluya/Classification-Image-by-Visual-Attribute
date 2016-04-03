import os
import numpy as np
import tarfile

def randomChoose(name):
	allImage = os.listdir('/Users/yuanluyao/Downloads/to_be_found/' + name)
	size = min(len(allImage), 100)
	chosen = list(np.random.permutation(len(allImage))[0: size])
	os.mkdir('/Users/yuanluyao/Downloads/to_be_found/' + name + '_chosen')
	for i in chosen:
		os.rename('/Users/yuanluyao/Downloads/to_be_found/' + name + '/' + allImage[i], '/Users/yuanluyao/Downloads/to_be_found/' + name + '_chosen/' + allImage[i])
	if size != 100:
		print name, size, 'not enough!'


#untar all the files
def main():
	
	f = open('names.txt', 'r')
	names = f.readlines()
	names = [name[0: -5] for name in names]
	''''
	for name in names:
		openfile = tarfile.open(name, 'r:')
		openfile.extractall(name[0: -4])
	'''
	for name in names:
		randomChoose(name)



if __name__ == '__main__':
	main()
