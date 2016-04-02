import xml.etree.ElementTree as et
import os



id = 'n01443537'
address = './bounding_box'+'/'+id+'/'+'Annotation'+'/'+id




def createTxt(path,id):
	res = open('./validImg'+'/'+id+'.txt','w')
	dirs = os.listdir(path)
	count = 0
	tot = 0
	#mmax = 0
	#mmin = 1
	c = 0
	dist = {}
	distxmin = {}
	distymin = {}
	distxmax = {}
	distymax = {}
	for file in dirs:		
		tree = et.parse(path+'/'+file)
		root = tree.getroot()
		size = root.find('size')
		ob = root.findall('object')
		if len(ob)==1:
			
			w = int(size.find('width').text)
			h = int(size.find('height').text)
			size = w*h
			bnd = root.find('object').find('bndbox')
			xmax = int(bnd.find('xmax').text)
			xmin = int(bnd.find('xmin').text)
			ymax= int(bnd.find('ymax').text)
			ymin = int(bnd.find('ymin').text)
			bnd_w = xmax-xmin
			bnd_h = ymax-ymin
			b_size = bnd_w*bnd_h
			tot  = tot+float(b_size)/size
			count  = count+1
			dist[file] = float(b_size)/size
			distxmin[file] = xmin
			distymin[file] = ymin
			distxmax[file] = xmax
			distymax[file] = ymax
	print count,id
	dist = sorted(dist.items(), key = lambda x: abs(0.5 - float(x[1])))
	dist = [element[0] for element in dist]
	for i in range(min(100, len(dist))):
		res.write(dist[i].split('.')[0]+' '+str(distxmin[dist[i]])+' '+str(distymin[dist[i]])+' '+str(distxmax[dist[i]])+' '+str(distymax[dist[i]])+'\n')


direct = './bounding_box'+'/'
dirs = os.listdir(direct)
for id in dirs:
	if not id == '.DS_Store':
		address = './bounding_box'+'/'+id+'/'+'Annotation'+'/'+id
		createTxt(address,id)
	

#createTxt(address,id)