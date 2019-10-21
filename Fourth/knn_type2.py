import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt;
from matplotlib import cm


df = pd.read_csv('Data_csv.csv',sep=",",header = None);

arr = df.values;

df2 = pd.read_csv('test_csv.txt',sep=",",header = None);

test = df2.values;

w1x = [] #class1 x_values
w1y = [] #class1 y_values
w2x = [] #class2 x_values
w2y = [] #class2 y_values

for i in range(len(arr)):
    if(arr[i][2] == 1):
        w1x.append(arr[i][0])
        w1y.append(arr[i][1])
    elif (arr[i][2] == 2):
        w2x.append(arr[i][0])
        w2y.append(arr[i][1])

#K = input("Enter value of K: ");

#print(K)

list = []

list_f = np.empty((0,4), int)

for i in range(len(test)):
    list2 = []
    class_ = []
    for j in range(len(arr)):

            val1 = np.power((test[i][0] - arr[j][0]), 2);
            val2 = np.power((test[i][1] - arr[j][1]), 2);
            distance = np.sqrt(val1 + val2)
            # print("Distance for ",test[i],": is ",val3)
            list2.append(distance)
            class_.append(arr[j][2])
    #list.append(list2)
    list_f = np.append(list_f, np.array([[test[i][0], test[i][1], list2, class_]]), 0)


predicted_list = []
k = int(input("Enter K value: "));


def bubbleSort(arr,arr2):
	n = len(arr)
	for i in range(n):
		for j in range(0, n-i-1):
			if arr[j] > arr[j+1] :
				arr[j], arr[j+1] = arr[j+1], arr[j]
				arr2[j], arr2[j+1] = arr2[j+1], arr2[j]


list_f2 = np.empty((0,3), int)

for i in range(len(list_f)):
    bubbleSort(list_f[i][2], list_f[i][3])
    sort_dis = list_f[i][2];
    sort_class = list_f[i][3];
    c1, c2 = 0, 0
    for j in range(k):
        if (sort_class[j] ==1):
            c1 = c1+1;
        else:
            c2 = c2+1;
    if c1>c2:
        list_f2 = np.append(list_f2, np.array([[list_f[i][0], list_f[i][1], 1]]), 0)
        predicted_list.append(1)
    else:
        list_f2 = np.append(list_f2, np.array([[list_f[i][0], list_f[i][1], 2]]), 0)
        predicted_list.append(2)

f = open("prediction.txt", "w")
for i in range(len(list_f)):
    f.write("Text point: %d, %d\n"%(list_f[i][0],list_f[i][1]));
    dis = list_f[i][2];
    cls = list_f[i][3];
    for j in range(k):
        f.write("Distance %d: %f \t Class: %d \n"%((j+1),dis[j],cls[j]));
        #list2)

    f.write("Predicted class:%d \n\n"%(predicted_list[i]))

f.close()


w1x_new = [] #class1 x_values
w1y_new = [] #class1 y_values
w2x_new = [] #class2 x_values
w2y_new = [] #class2 y_values

for i in range(len(list_f2)):
    if(list_f2[i][2] == 1):
        w1x_new.append(list_f2[i][0])
        w1y_new.append(list_f2[i][1])
    elif(list_f2[i][2] == 2):
        w2x_new.append(list_f2[i][0])
        w2y_new.append(list_f2[i][1])

print("predicted:")
print(predicted_list)
print("Actual:")
print(test[:,2])

lenght_actual = len(test[:,2])

tp,fn,fp,tn = 0,0,0,0

for i in range(lenght_actual):
    if predicted_list[i]==1 and test[i,2]==1:
        tp = tp +1;
    elif predicted_list[i]==1 and test[i,2]==2:
        fp = fp + 1;
    elif predicted_list[i]==2 and test[i,2]==1:
        fn = fn + 1;
    elif predicted_list[i]==2 and test[i,2]==2:
        tn = tn + 1;


print("TP:",tp)
print("TN:",tn)
print("FP:",fp)
print("FN:",fn)

print("Recall:",(tp/(tp+fn)))
print("precision:",(tp/(tp+fp)))
print("accuracy:",((tp+tn)/(tp+tn+fp+fn)))

pre = tp/(tp+fp)
recall = tp/(tp+fn)

total_ = (1/pre)+(1/recall)

print("f_measure:",(2/total_))


plt.plot(w1x,w1y,'^r',label='class1 train');
plt.plot(w2x,w2y,'ob',label='class2 train');
plt.plot(w1x_new,w1y_new,'*g',label='Classified class1 train');
plt.plot(w2x_new,w2y_new,'+y',label='classified class2 train');

plt.title('Implementing K-Nearest Neighbors (KNN)')
plt.legend();
plt.show();
