import os
import subprocess
import tarfile
import sys

def main():
	destinationFile = '../data_with_bounding_box/'
	os.mkdir(destinationFile)
	folderNames = os.listdir('.')
	fileName = sys.argv[1]
	directory = '../' + fileName + '/'
	
	#unzip file
	for folderName in folderNames:
		openfile = tarfile.open(directory + folderName[0: -4] + '.tar', 'r:')
		openfile.extractall(directory + folderName[0: -4])
	
	#read bounding box file
	for folderName in folderNames:
		f = open(folderName, 'r')
		lines = f.readlines()
		for line in lines:
			line = line.split()
			name = line[0]
			xmin = int(line[1])
			ymin = int(line[2])
			xmax = int(line[3])
			ymax = int(line[4])
			xlen = xmax - xmin
			ylen = ymax - ymin
			#first crop
			subprocess.call('convert -crop ' + directory + folderName[0: -4] + '/' + name + '.JPEG ' + str(xlen) + 'x' + str(ylen) + '+' + line[1] + '+' + line[2] + ' ' + directory + folderName[0: -4] + '/' + name + '.JPEG', shell = True)
			subprocess.call('convert ' + directory + folderName[0: -4] + '/' + name + '.JPEG ' + '-resize 256x256 ' + destinationFile + '/' + name + '.JPEG', shell = True)


if __name__ == '__main__':
	main()