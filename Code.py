import xlrd
import sympy
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy as sp
from scipy.optimize import leastsq

def func(p,x):
	a,b,c=p
	return a*(x**b)+c

def error(p,x,y):
	return func(p,x)-y

def HER2():
	position='/Users/songhaoqiu/Desktop/HER2.xlsx'
	result=[]
	table=xlrd.open_workbook(position).sheets()[0]
	for i in range(1,table.nrows):
		col=table.row_values(i)
		result.append(col)
	X=[]
	Y=[]
	for i in range(0,table.nrows-1):
		temp=[]
		temp.append(result[i][2])
		temp.append(result[i][3])
		temp.append(result[i][4])
		X.append(temp)
		temp2=[result[i][13]]
		Y.append(temp2)
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
	knn = KNeighborsClassifier()
	aaa=knn.fit(x_train,y_train)
	return aaa

def LBP():
	position='/Users/songhaoqiu/Desktop/LBP.xlsx'
	result=[]
	table=xlrd.open_workbook(position).sheets()[0]
	for i in range(1,table.nrows):
		col=table.row_values(i)
		result.append(col)
	X=[]
	Y=[]
	for i in range(0,table.nrows-1):
		temp=[]
		temp.append(result[i][2])
		temp.append(result[i][5])
		temp.append(result[i][7])
		X.append(temp)
		temp2=[result[i][13]]
		Y.append(temp2)
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
	xgbc = XGBClassifier()
	X_train=np.array(x_train)
	X_test=np.array(x_test)
	aaa=xgbc.fit(X_train,y_train)
	return aaa

def LBN():
	position='/Users/songhaoqiu/Desktop/LBN.xlsx'
	result=[]
	table=xlrd.open_workbook(position).sheets()[0]
	for i in range(1,table.nrows):
		col=table.row_values(i)
		result.append(col)
	X=[]
	Y=[]
	for i in range(0,table.nrows-1):
		temp=[]
		temp.append(result[i][4])
		temp.append(result[i][7])
		X.append(temp)
		temp2=[result[i][13]]
		Y.append(temp2)
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
	xgbc = XGBClassifier()
	X_train=np.array(x_train)
	X_test=np.array(x_test)
	aaa=xgbc.fit(X_train,y_train)
	return aaa

def TN():
	position='/Users/songhaoqiu/Desktop/TN.xlsx'
	result=[]
	table=xlrd.open_workbook(position).sheets()[0]
	for i in range(1,table.nrows):
		col=table.row_values(i)
		result.append(col)
	X=[]
	Y=[]
	for i in range(0,table.nrows-1):
		temp=[]
		temp.append(result[i][4])
		temp.append(result[i][5])
		X.append(temp)
		temp2=[result[i][13]]
		Y.append(temp2)
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
	rf = RandomForestClassifier()
	aaa=rf.fit(x_train,y_train)
	return aaa

import pickle
h=HER2()
bp=LBP()
bn=LBN()
t=TN()
data_output1 = open('/Users/songhaoqiu/Desktop/HER2.pkl','wb')
pickle.dump(h,data_output1)
data_output1.close()

data_output2 = open('/Users/songhaoqiu/Desktop/LBP.pkl','wb')
pickle.dump(bp,data_output2)
data_output2.close()

data_output3 = open('/Users/songhaoqiu/Desktop/LBN.pkl','wb')
pickle.dump(bn,data_output3)
data_output3.close()

data_output4 = open('/Users/songhaoqiu/Desktop/TN.pkl','wb')
pickle.dump(t,data_output4)
data_output4.close()

"""
import sympy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy as sp
from scipy.optimize import leastsq
import pickle

intro=input()
if(intro==""):
	print("When using this program, first input a subtype name, including “HER2”, “LBP”, “LBN”, and “TN”. Then input the total area of the tumors after the first, second, and third chemotherapies. As for the four parameters, each parameter is separated by a space character.")
else:
	sub,bb,cc,dd=intro.split()
	b=float(bb)
	c=float(cc)
	d=float(dd)
	b=b+1
	c=c+1
	d=d+1
	read=[]
	read.append(b)
	read.append(c)
	read.append(d)
	
	def func(p,x):
		a,b,c=p
		return a*(x**b)+c
	
	def error(p,x,y):
		return func(p,x)-y
	
	def HER2():
		aa=[]
		aa.append(read)
		b=read_data.predict(aa)
		bbb = str(b)
		bbb = bbb.replace('[','')
		bbb = bbb.replace(']','')
		bbb = bbb.replace('\'','')
		return bbb
	
	def LBP():
		Xi=np.array([1,2,3])
		Yi=np.array(read)
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time4=a*(4**b)+c
		
		Xi=np.array([2,3,4])
		Yi=np.array([read[1],read[2],time4])
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time5=a*(5**b)+c
		
		Xi=np.array([3,4,5])
		Yi=np.array([read[2],time4,time5])
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time6=a*(6**b)+c
		
		final=[]
		final.append(read[0])
		final.append(time4)
		final.append(time6)
		
		aa=[]
		aa.append(final)
		b=read_data.predict(aa)
		bbb = str(b)
		bbb = bbb.replace('[','')
		bbb = bbb.replace(']','')
		bbb = bbb.replace('\'','')
		return bbb
	
	def LBN():
		Xi=np.array([1,2,3])
		Yi=np.array(read)
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time4=a*(4**b)+c
		
		Xi=np.array([2,3,4])
		Yi=np.array([read[1],read[2],time4])
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time5=a*(5**b)+c
		
		Xi=np.array([3,4,5])
		Yi=np.array([read[2],time4,time5])
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time6=a*(6**b)+c
		
		final=[]
		final.append(read[2])
		final.append(time6)
		
		aa=[]
		aa.append(final)
		b=read_data.predict(aa)
		bbb = str(b)
		bbb = bbb.replace('[','')
		bbb = bbb.replace(']','')
		bbb = bbb.replace('\'','')
		return bbb
	
	def TN():
		Xi=np.array([1,2,3])
		Yi=np.array(read)
		p0=[10,10,10]
		Para=leastsq(error,p0,args=(Xi,Yi),maxfev=500000)
		a,b,c=Para[0]
		time4=a*(4**b)+c
		
		final=[]
		final.append(read[2])
		final.append(time4)
		
		aa=[]
		aa.append(final)
		b=read_data.predict(aa)
		bbb = str(b)
		bbb = bbb.replace('[','')
		bbb = bbb.replace(']','')
		bbb = bbb.replace('\'','')
		return bbb
	
	if(sub=="HER2"):
		data_input = open('/Users/songhaoqiu/Desktop/HER2.pkl','rb')
		read_data = pickle.load(data_input)
		data_input.close()
		print(HER2())
	elif(sub=="LBP"):
		data_input = open('/Users/songhaoqiu/Desktop/LBP.pkl','rb')
		read_data = pickle.load(data_input)
		data_input.close()
		print(LBP())
	elif(sub=="LBN"):
		data_input = open('/Users/songhaoqiu/Desktop/LBN.pkl','rb')
		read_data = pickle.load(data_input)
		data_input.close()
		print(LBN())
	elif(sub=="TN"):
		data_input = open('/Users/songhaoqiu/Desktop/TN.pkl','rb')
		read_data = pickle.load(data_input)
		data_input.close()
		print(TN())
	else:
		print("Subtype input error!\nOnly the four subtypes “HER2”, “LBP”, “LBN”, and “TN” is acceptable.")
"""
