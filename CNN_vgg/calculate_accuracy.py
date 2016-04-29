import json

f = open("labels_name_softmax.json", 'r')
labels = json.loads(f.readline())
f = open("predications_name_softmax.json", 'r')
predication = json.loads(f.readline())
dict = {}
for name in predication.keys():
	if dict.get(name[0: 9]):
		dict[name[0: 9]][0] += (float(str(labels[name])) in eval(str(predication[name])))
		dict[name[0: 9]][1] += 1
	else:
		dict[name[0: 9]] = [0 + (predication[name] == labels[name]), 1]
f = open('accuracy_softmax.txt', 'w')
for clas in dict.keys():
	f.write(clas + " " + str(dict[clas][0]) + " " + str(dict[clas][1]) + " " + str(1.0 * dict[clas][0] / dict[clas][1]) + '\n')
