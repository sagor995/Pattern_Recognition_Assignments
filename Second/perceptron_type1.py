import numpy as np;
import pandas as pd;
import random as rnd;
import matplotlib.pyplot as plt;
from math import pow;

np.random.seed(0)

#df = pd.read_csv('train2.txt', sep=" ", header=None, dtype='float64')
df = pd.read_csv('train.txt', sep=" ", header=None)

#print(df)

arr = df.values

print(type(arr))

w1x, w1y, w2x, w2y=[], [], [], []

for i in range(len(arr)):
    if arr[i][2] == 1.0:
        w1x.append(arr[i][0])
        w1y.append(arr[i][1])
    else:
        w2x.append(arr[i][0])
        w2y.append(arr[i][1])

#x square = x1_2
#y square = x2_2

y=np.empty((0,6), int)


for i in range(len(arr)):
    if arr[i][2] == 1.0:
        x1_2 = arr[i][0] * arr[i][0]
        x2_2 = arr[i][1] * arr[i][1]
        x1_x2 = arr[i][0] * arr[i][1];
        #y.append([x1_2, x2_2, x1_x2, arr[i][0], arr[i][1], 1])
        #y = np.append(y,[x1_2, x2_2, x1_x2, arr[i][0], arr[i][1], 1])
        y = np.append(y, np.array([[x1_2, x2_2, x1_x2, arr[i][0], arr[i][1], 1]]), 0)
    else:
        x1_2 = -(arr[i][0] * arr[i][0])
        x2_2 = -(arr[i][1] * arr[i][1])
        x1_x2 = -(arr[i][0] * arr[i][1]);
        #y = np.append(y, [x1_2, x2_2, x1_x2, arr[i][0], arr[i][1], 1])
        #y.append([x1_2, x2_2, x1_x2, -arr[i][0], -arr[i][1], -1]);
        y = np.append(y, np.array([[x1_2, x2_2, x1_x2, -arr[i][0], -arr[i][1], -1]]), 0)

print("Y is: ",y)
print("\n")

w_one=np.ones((1, 6),dtype = int)
w_zero=np.zeros((1, 6),dtype = int)
w_rand=np.random.random((1,6))

print("When w=1: ",w_one);
print("When w=0: ",w_zero);
print("When w=rand: ",w_rand);

alpha_list = []

one_at_a_time1 = [];
many_at_a_time1 = [];

one_at_a_time0 = [];
many_at_a_time0 = [];

one_at_a_time2 = [];
many_at_a_time2 = [];

def one_at_a_time_fun(w, alpha):
    g = 0;
    check = 0;
    count = 0;

    #-One at a time-
    while check != 6:
        count = count + 1;
        check = 0;
        for i in range(6):
            g = np.dot(y[i, :], w.transpose());
            if g <= 0:
                w = w + alpha * y[i, :];
            else:
                check = check + 1;
        if check == 6:
            break;

    return count;

def many_at_a_time_fun(w, alpha):
    g = 0;
    check = 0;
    count = 0;
    w_new = 0;

    #-Many at a time-
    while check != 6:
        count = count + 1;
        check = 0;
        for i in range(6):
            g = np.dot(y[i, :], w.transpose());
            if g <= 0:
                w_new = w_new + y[i, :];
            else:
                check = check + 1;
        if check == 6:
            break;
        w = w + alpha * w_new;

    return count;



#for one
for i in range(1,11,1):
    alpha_list.append(float(i / 10))
    one_at_a_time1.append(one_at_a_time_fun(w_one,float(i/10)))

for i in range(1, 11, 1):
    many_at_a_time1.append(many_at_a_time_fun(w_one, float(i / 10)))


#for zero
for i in range(1,11,1):
    one_at_a_time0.append(one_at_a_time_fun(w_zero,float(i/10)))

for i in range(1, 11, 1):
    many_at_a_time0.append(many_at_a_time_fun(w_zero, float(i / 10)))

#for rand
for i in range(1,11,1):
    one_at_a_time2.append(one_at_a_time_fun(w_rand,float(i/10)))

for i in range(1, 11, 1):
    many_at_a_time2.append(many_at_a_time_fun(w_rand, float(i / 10)))


#print(alpha_list)
print("alpha values: ",alpha_list)
print("One at a time: w1 ",one_at_a_time1)
print("Many at a time: w1 ",many_at_a_time1)
print("")
print("One at a time: w0 ",one_at_a_time0)
print("Many at a time: w0 ",many_at_a_time0)
print("")
print("One at a time: wR ",one_at_a_time2)
print("Many at a time: wR ",many_at_a_time2)



plt.title('Perceptron algorithm for finding the weights.')
plt.plot(w1x,w1y,'*r',label='w1 class')
plt.plot(w2x,w2y,'ob',label='w2 class')

plt.legend();
plt.show();






#Bar chat for 1
bar_width = 0.02
rects1 = plt.bar([x-0.01 for x in alpha_list], one_at_a_time1, bar_width,
color='b',
label='One at a time', align='center')

rects2 = plt.bar([x+0.01 for x in alpha_list], many_at_a_time1, bar_width,
color='r',
label='Many at a time', align='center')

plt.xlabel('Number of iterations.')
plt.ylabel('Learning rate.')
plt.title('Perceptron algorithm-All w one')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()

#Bar chat for 0
bar_width = 0.02
rects1 = plt.bar([x-0.01 for x in alpha_list], one_at_a_time0, bar_width,
color='b',
label='One at a time', align='center')

rects2 = plt.bar([x+0.01 for x in alpha_list], many_at_a_time0, bar_width,
color='r',
label='Many at a time', align='center')

plt.xlabel('Number of iterations')
plt.ylabel('Learning rate')
plt.title('Perceptron algorithm-All w zero')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()


#Bar chat for rand
bar_width = 0.02
rects1 = plt.bar([x-0.01 for x in alpha_list], one_at_a_time2, bar_width,
color='b',
label='One at a time')

rects2 = plt.bar([x+0.01 for x in alpha_list], many_at_a_time2, bar_width,
color='r',
label='Many at a time')

plt.xlabel('Number of iterations')
plt.ylabel('Learning rate')
plt.title('Perceptron algorithm-All w randoms')
plt.xticks(alpha_list, alpha_list)
plt.legend()
plt.show()
