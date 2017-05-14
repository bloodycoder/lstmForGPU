import numpy as np
output_file_name = 'output_lstm'
f = open('./output/'+output_file_name,'r')
data = f.read().split('\n')
for i in range(33,len(data)):
	if(len(data[i])<5):
		data = data[:i]
		break;
for i in range(0,33):
	data[i] = data[i].split(' ')
np.array(data)
print(data)
# coe is data[:,1]
