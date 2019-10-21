import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#task 1
df = pd.read_csv('train.txt', sep=" " ,  header = None, dtype = 'Int64')
row, col = df.shape
class1 = []
class2 = []
for i in range(row):
    temp = []
    for j in range(col):
        temp.append(df[j][i])
    if df[col - 1][i] == 1:
        class1.append(temp)
    else:
        class2.append(temp)
class1 = np.array(class1)
class2 = np.array(class2)
class1_t = np.transpose(class1)
class2_t = np.transpose(class2)
plt.scatter(class1_t[0], class1_t[1], color = 'red', marker = '*')
plt.scatter(class2_t[0], class2_t[1], color = 'blue', marker = '^')
plt.show()

#task 2
class1_mean = np.array([np.mean(class1_t[0], dtype='Int64'), np.mean(class1_t[1], dtype='Int64')])
class2_mean = np.array([np.mean(class2_t[0], dtype='Int64'), np.mean(class2_t[1], dtype='Int64')])
df_test = pd.read_csv('test.txt', sep=" " ,  header = None, dtype = 'Int64')
row_test, col_test = df_test.shape
test_class1 = []
test_class2 = []
for i in range(row_test):
    X = np.array([df_test[0][i], df_test[1][i]])
    g1 = X.dot(np.transpose(class1_mean)) - (int((0.5 * class1_mean.dot(np.transpose(class1_mean)))))
    g2 = X.dot(np.transpose(class2_mean)) - (int((0.5 * class2_mean.dot(np.transpose(class2_mean)))))
    if g1 > g2:
        X = np.append(X,1)
        test_class1.append(X)
    else:
        X = np.append(X,2)
        test_class2.append(X)
test_class1_t = np.transpose(test_class1)
test_class2_t = np.transpose(test_class2)
plt.scatter(class1_t[0], class1_t[1], color = 'red', marker = '*')
plt.scatter(class2_t[0], class2_t[1], color = 'blue', marker = '^')
plt.scatter(test_class1_t[0], test_class1_t[1], color = 'red', marker = 'o')
plt.scatter(test_class2_t[0], test_class2_t[1], color = 'blue', marker = 'p')
plt.show()

#task 3
c = int(0.5 * (class1_mean.dot(np.transpose(class1_mean)) - class2_mean.dot(np.transpose(class2_mean))))
m = class1_mean - class2_mean
m1 = m[0]
m2 = m[1]
y = []
x = [i for i in range(-10,10,1)]
for i in range(len(x)):
    y.append((int(((x[i] * m1) + c) / m2)) * (-1))
plt.scatter(class1_t[0], class1_t[1], color = 'red', marker = '*')
plt.scatter(class2_t[0], class2_t[1], color = 'blue', marker = '^')
plt.scatter(test_class1_t[0], test_class1_t[1], color = 'red', marker = 'o')
plt.scatter(test_class2_t[0], test_class2_t[1], color = 'blue', marker = 'p')
plt.plot(x, y, color = 'blue')
plt.show()
