import numpy as np;
import pandas as pd;
import random as rnd;
import matplotlib.pyplot as plt;
from math import pow;

np.random.seed(0)

df = pd.read_csv('iris.data', sep=",", header=None)

arr = df.values

print(type(arr))

w1x, w1y, w2x, w2y=[], [], [], []

for i in range(len(arr)):
    if arr[i][4] == 'Iris-setosa':
        w1x.append(arr[i][0])
        w1y.append(arr[i][2])
    elif arr[i][4] == 'Iris-versicolor':
        w2x.append(arr[i][0])
        w2y.append(arr[i][2])

print(w1x)
print(w2x)

y=np.empty((0,6), int)

for i in range(100):
    if arr[i][4] == 'Iris-setosa':
        x1_2 = arr[i][0] * arr[i][0]
        x3_2 = arr[i][2] * arr[i][2]
        x1_x3 = arr[i][0] * arr[i][2];
        y = np.append(y, np.array([[x1_2, x3_2, x1_x3, arr[i][0], arr[i][2], 1]]), 0)
    elif arr[i][4] == 'Iris-versicolor':
        x1_2 = -(arr[i][0] * arr[i][0])
        x3_2 = -(arr[i][2] * arr[i][2])
        x1_x3 = -(arr[i][0] * arr[i][2]);
        y = np.append(y, np.array([[x1_2, x3_2, x1_x3, -arr[i][0], -arr[i][2], -1]]), 0)

print(y)
print("\n")

w_one=np.ones((1, 6),dtype = int)

print("When w=1: ", w_one.transpose());

alpha_list = []

one_at_a_time1 = [];



def one_at_a_time_fun(w, alpha):
    g = 0;
    check = 0;
    count = 0;

    while check != 100:
        count = count + 1;
        check = 0;
        for i in range(100):
            g = np.dot(y[i, :], w.transpose());
            if g <= 0:
                w = w + alpha * y[i, :];
            else:
                check = check + 1;
        if check == 100:
            break;

    return count;


#for one
for i in range(1,11,1):
    alpha_list.append(float(i / 10))
    one_at_a_time1.append(one_at_a_time_fun(w_one,float(i/10)))


#print(alpha_list)
print("alpha values: ",alpha_list)
print("One at a time: w1 ",one_at_a_time1)


plt.title('Perceptron algorithm for finding the weights.')
plt.plot(w1x,w1y,'*r',label='w1 class')
plt.plot(w2x,w2y,'ob',label='w2 class')

plt.legend();
plt.show();


#Bar chat for 1
bar_width = 0.02
rects1 = plt.bar(alpha_list, one_at_a_time1, bar_width,
color='b',
label='One at a time', align='center')

plt.xlabel('Number of iterations.')
plt.ylabel('Perceptron algorithm-All w one')
plt.title('Learning rate.')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()
