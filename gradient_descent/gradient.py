""" 
Author @SwApNiL
Gradient Descent for multivariable
"""
import pandas as pd 
import numpy as np 
import copy
from collections import OrderedDict

#extract data as dictionary using pandas 
dat = pd.read_csv('train.csv').to_dict()
y = dat['price']
# incase if dat -- orignal is needed
data = copy.deepcopy(dat)
# delete entries from dictionary
del data['price']
del data['id']
# no. of samples/instances
data_length = len(y)

#calculation of predicted value using w vector b and features
def regModel(w,b,points):
	# instead of 'view' original file can provide lenght for vector
	summ = [0]*len(points['view'])
	for i in range(0,data_length):
		for k in points:
			summ[i] += points[k][i]*w[k]
		summ[i] += b
	return summ 

# MSE calc.
def error(w,b,points,y):
	totalE = 0
	predicted = regModel(w,b,points)
	for i in range(0,data_length):
		totalE += (y[i] - predicted[i]) ** 2 
	return totalE/float(data_length) 	

# calc. of gradient descent
def gradient(w_current,b_current,points,y,n):  #n is learning rate
	w_g = dict()
	for i in data:
		w_g[i] = 0
	b_g = 0

	divr = regModel(w_current,b_current,points)
	for k in points:
		for i in range(0,data_length):
			#gradient of MSE wrt to w's
			w_g[k] += - (2/float(data_length))*points[k][i]*(y[i]- divr[i])
		w_new[k] = w_current[k] -(n*w_g[k])

	for i in range(0,data_length):
		#gradient of MSE wrt to b		
		b_g += - (2/float(data_length))*(y[i]- divr[i])
	b_new = b_current + (n*b_g)
	return w_new, b_new

#normalization of y vector/output
def out_norm(vect):
	mean=0
	for i in range(len(vect)):
		mean += vect[i]
	mean=mean/float(len(vect))
	var=0
	for i in range(len(vect)):
		var += (vect[i]-mean) ** 2
	var = var/float(len(vect))
	#print mean,var
	if var!=0:
		for i in range(data_length):
			vect[i] = (vect[i] - mean)/(var) ** (0.5)
	else:
		for i in range(data_length):
			vect[i] = 0
	#taking outputs mean and var as well because need for scaling back again
	return vect,mean,var

# normalization of features
def normalization(vect):
	mean=0
	for i in range(len(vect)):
		mean += vect[i]
	mean=mean/float(len(vect))
	var=0
	for i in range(len(vect)):
		var += (vect[i]-mean) ** 2
	var = var/float(len(vect))
	#print mean,var
	if var!=0:
		for i in range(len(vect)):
			vect[i] = (vect[i] - mean)/(var) ** (0.5)
	else:
		for i in range(len(vect)):
			vect[i] = 0

	return vect
# for testing of dataframes	
def predictor(w,b,data,y_mean,y_var):
	y = [0]*len(data['view'])
	for i in range(0,len(y)):
		for k in data:
			#print y[i],k
			y[i] += w[k]*data[k][i]
		y[i] = y[i] + b
		y[i] = y[i]*(y_var) ** (0.5) + y_mean
	return y

# initializations
w_new = dict()
for i in data:
	w_new[i] = 0
b_new = 0
y_mean =0
y_var = 1
# data norm
y_norm,y_mean,y_var = out_norm(y)
# feature norms
for k in data:
	data[k] = normalization(data[k])
	#print k, data[k]
# iteration for error minimization
for itr in range(0,100):
	w_new,b_new = gradient(w_new,b_new,data,y_norm,0.1)
	Error = error(w_new,b_new,data,y_norm)
	print itr, Error 

# gained w's and b
print w_new,b_new

#-------------------reading testing dataframes -------------
dat_read = pd.read_csv('test.csv').to_dict()
#from dict to values
ids =  dat_read['id'].values()

data_read = copy.deepcopy(dat_read)
del data_read['id']
for k in data_read:
	data_read[k] = normalization(data_read[k])

y_predicted = predictor(w_cal,b_cal,data_read,y_mean,y_var)
# raw_data is used for generating output file
raw_data = dict()
# adding entries in dict
raw_data['id'] = dat_read['id']
raw_data['price'] = y_predicted
#print y_predicted

# OrderedDict provides order list of dict for conversion to pandas dataframe
raw_data = OrderedDict([('id',ids),('price',y_predicted)]) 
# taking orderedDict as input and converting pandas dataframe
df = pd.DataFrame.from_dict(raw_data)
# seperated by tab and use encoding for linux
df.to_csv('out.csv', sep='\t', encoding='utf-8')
