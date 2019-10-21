import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import math as m;

df = pd.read_csv('train.txt',sep=" ",header = None,dtype='int64');

arr = df.values;


w1x = [] #class1 x_values
w1y = [] #class1 y_values
w2x = [] #class2 x_values
w2y = [] #class2 y_values

for i in range(len(arr)):
    if(arr[i][2] == 1):
        w1x.append(arr[i][0])
        w1y.append(arr[i][1])
    else:
        w2x.append(arr[i][0])
        w2y.append(arr[i][1])

#Mean of 2 class

w1_mean = [];
w2_mean = [];

w1_mean.append(np.mean(w1x));
w1_mean.append(np.mean(w1y));

w2_mean.append(np.mean(w2x));
w2_mean.append(np.mean(w2y));



#print(w1_mean)

"""
w1_mean_x = np.mean(w1x);
w1_mean_y = np.mean(w1y);
w2_mean_x = np.mean(w2x);
w2_mean_y = np.mean(w2y);
"""


#Test File
df1 = pd.read_csv('test.txt',sep=" ",header = None,dtype='int64');

testFile = df1.values;
tx = []


for i in range(len(testFile)):
    tx.append(testFile[i][0])
    tx.append(testFile[i][1])

#print(tx)

length = len(tx)/2
text = np.array(tx).reshape(2, int(length))

text = np.matrix(text).T#Tranposing input test

class1_testX = []
class1_testY = []
class2_testX = []
class2_testY = []


#for accuracy checking

classCounter = []
matchCounter = 0;




w1_mean_matrix = np.matrix(w1_mean);
w2_mean_matrix = np.matrix(w2_mean);



#calculating linear discimination function
for i in range(len(text)):
    g1 = np.dot(text[i,:],w1_mean_matrix.T) - (.5*w1_mean_matrix.dot(w1_mean_matrix.T))
    g2 = np.dot(text[i,:],w2_mean_matrix.T) - (.5*w2_mean_matrix.dot(w2_mean_matrix.T))
    #print(g1,"____",g2)


    if(g1>g2):
        classCounter.append(1);
        class1_testX.append(text[i,0])
        class1_testY.append(text[i,1])
        #plt.plot(text[i,0],text[i,1],'<r',label='Class1 Test');
    else:
        classCounter.append(2);
        class2_testX.append(text[i, 0])
        class2_testY.append(text[i, 1])
        #plt.plot(text[i, 0], text[i, 1], '>b', label='Class2 Test');


for i in range(len(testFile)):
    if testFile[i][2] == classCounter[i]:
        matchCounter += 1;

print((matchCounter/len(testFile))*100,'%');

plt.plot(w1x,w1y,'*r',label='class1 train');
plt.plot(w2x,w2y,'ob',label='class2 train');

plt.plot(w1_mean[0],w1_mean[1],'+r',label='class1 mean');
plt.plot(w2_mean[0],w2_mean[1],'_b',label='class2 mean');

plt.plot(class1_testX,class1_testY,'<r',label='class1 test');
plt.plot(class2_testX,class2_testY,'>b',label='class2 test');

#plt.plot(class2_testX,class2_testY,'.-g',label='DB');
plt.title('Minimum distance to class mean classifier')
plt.legend();
plt.show();
