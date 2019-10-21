import numpy as np
import pandas as pd
import math
import random as rnd
import matplotlib.pyplot as plt
from math import pow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt;
from matplotlib import cm

sd1 = np.array([[.25, .3], [.3, 1.0]])
sd2 = np.array([[.5, 0], [0, .5]])


meu1 = np.array([0, 0])
meu2 = np.array([2, 2])

df = pd.read_csv('tesst.txt', sep=",", header=None)
arr = df.values

x=np.empty((0,2), int)

for i in range(len(arr)):
    x = np.append(x, np.array([[arr[i][0], arr[i][1]]]), 0)


n1_value = []
n2_value = []

def normalDistribution(x,mean,sd,w):
    m=2;
    sd_inv = np.linalg.inv(sd)
    xu_norm = (x - mean)
    p1 = sd_inv.dot(np.transpose(xu_norm))
    f1 = 0.5 * (xu_norm .dot(p1))
    f2 = (m / 2) *np.log(2 * np.pi);
    sd_det = np.linalg.det(sd);
    f3 = 0.5*np.log(sd_det);
    total = -f1-f2-f3;
    return np.exp(total+np.log(w))

for i in range(len(x)):
    value = normalDistribution(x[i], meu1, sd1,0.5);
    print(value)
    n1_value.append(value);

print("Values are: ",n1_value)

for i in range(len(x)):
    value = normalDistribution(x[i], meu2, sd2,0.5);
    print(value)
    n2_value.append(value);


class1_valueX = []
class1_valueY = []
class2_valueX = []
class2_valueY = []


n_dff = []

for i in range(len(arr)):
    if n1_value[i] > n2_value[i]:
        n_dff.append(n1_value[i]-n2_value[i])
        class1_valueX.append(arr[i][0])
        class1_valueY.append(arr[i][1])
    elif n2_value[i] > n1_value[i]:
        n_dff.append(n2_value[i] - n1_value[i])
        class2_valueX.append(arr[i][0])
        class2_valueY.append(arr[i][1])


print("Diff: ",n_dff)

print("N1: ",n1_value)
print("N2: ",n2_value)

print("value separating: ")
print("class1 x", class1_valueX)
print("class1 y", class1_valueY)
print("class2 x", class2_valueX)
print("class2 y", class2_valueY)

"""
plt.title('Implementing Minimum Error Rate Classifier.')
plt.plot(class1_valueX,class1_valueY,'*r',label='w1 class')
plt.plot(class2_valueX,class2_valueY,'ob',label='w2 class')
plt.legend();
plt.show();
"""

class1_valueZs = np.zeros(len(class1_valueX))
class2_valueZs = np.zeros(len(class2_valueX))

X1 = np.arange(np.min(class1_valueX)-1.8, np.max(class1_valueX), 0.05)
Y1 = np.arange(np.min(class1_valueY), np.max(class1_valueY), 0.05)

X1, Y1 = np.meshgrid(X1,Y1)

pos1 = np.empty(X1.shape + (2,))
pos1[:, :, 0] = X1
pos1[:, :, 1] = Y1

X2 = np.arange(np.min(class2_valueX), np.max(class2_valueX), 0.05)
Y2 = np.arange(np.min(class2_valueY), np.max(class2_valueY), 0.05)

X2, Y2 = np.meshgrid(X2,Y2)

pos2 = np.empty(X2.shape + (2,))
pos2[:, :, 0] = X2
pos2[:, :, 1] = Y2


def m_g(pos, mean, sd):
    n = mean.shape[0]
    sd_det = np.linalg.det(sd)
    sd_inv = np.linalg.inv(sd)
    N =np.sqrt(np.power(2 * np.pi, n) * sd_det)
    fac = np.einsum('ijk,kl,ijl->ij', pos-mean, sd_inv, pos-mean)
    return np.exp(-fac / 2) / N




Z1 = m_g(pos1, meu1, sd1)
Z2 = m_g(pos2, meu2, sd2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, Y1, Z1, rstride=3, cstride=3, antialiased=True, alpha=0.4, cmap=cm.ocean)
cset = ax.contour(X1, Y1, Z1, zdir='z', offset=-0.18, cmap=cm.ocean)

ax.plot_surface(X2, Y2, Z2, rstride=3, cstride=3, antialiased=True, alpha=0.4, cmap=cm.ocean)
cset = ax.contour(X2, Y2, Z2, zdir='z', offset=-0.18, cmap=cm.ocean)


ax.scatter(class1_valueX,class1_valueY,class1_valueZs,c='r',marker='*');
ax.scatter(class2_valueX,class2_valueY,class2_valueZs,c='b',marker='o');

#plt.plot(np.dot(class2_valueX,0.75)+np.dot(class2_valueX,-.025),np.dot(class2_valueY,0.75)+np.dot(class2_valueY,-.025),color='black', marker="*")

ax.set_zlabel("Probability Density");
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-0.15,0.4)
ax.view_init(30, -125)
plt.show()