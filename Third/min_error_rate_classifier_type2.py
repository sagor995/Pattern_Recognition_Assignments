import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt;
from matplotlib import cm

df = pd.read_csv('Data_csv.csv', sep=",", header=None)

arr = df.values

w1x, w1y, w2x, w2y=[], [], [], []

#print(arr)

m11,m12,m21,m22 = 0,0,0,0

for i in range(len(arr)):
    if arr[i][2] == 1.0:
        w1x.append(arr[i][0])
        w1y.append(arr[i][1])
    elif arr[i][2] == 2.0:
        w2x.append(arr[i][0])
        w2y.append(arr[i][1])


n11 =np.mean(w1x);
m12 = np.mean(w1y);

n21 =np.mean(w2x);
m22 = np.mean(w2y);

mean1 = np.array([m11,m12])
mean2 = np.array([m21,m22])

cov_mat1 = np.stack((w1x,w1y),axis=0)
cov_mat2 = np.stack((w2x,w2y),axis=0)


sigma1 = np.cov(cov_mat1)
sigma2 = np.cov(cov_mat2)

total = len(arr);
pw1 = len(w1x)/total;
pw2 = len(w1x)/total;




df2 = pd.read_csv('test_csv.txt', sep=",", header=None)
arr2 = df2.values

x=np.empty((0,2), int)

for i in range(len(arr2)):
    x = np.append(x, np.array([[arr2[i][0], arr2[i][1]]]), 0)

n1_value = []
n2_value = []


def n_d(x,mean,sd,w):
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
    value = n_d(x[i], mean1, sigma1,pw1);
    print(value)
    n1_value.append(value);

print("")

for i in range(len(x)):
    value = n_d(x[i], mean2, sigma2,pw2);
    print(value)
    n2_value.append(value);

class1_valueX = []
class1_valueY = []
class2_valueX = []
class2_valueY = []

for i in range(len(arr2)):
    if n1_value[i] > n2_value[i]:
        class1_valueX.append(arr2[i][0])
        class1_valueY.append(arr2[i][1])
    elif n2_value[i] > n1_value[i]:
        class2_valueX.append(arr2[i][0])
        class2_valueY.append(arr2[i][1])


print("N1: ",n1_value)
print("N2: ",n2_value)

print("value separating: ")
print("class1 x", class1_valueX)
print("class1 y", class1_valueY)
print("class2 x", class2_valueX)
print("class2 y", class2_valueY)


class1_valueZs = np.zeros(len(class1_valueX))
class2_valueZs = np.zeros(len(class2_valueX))

X5 = np.arange(0,12, 0.05);
Y5 = np.arange(0,12, 0.05);

X5, Y5 = np.meshgrid(X5, Y5);

pos5 = np.empty(X5.shape + (2,))
print("Postion: ",X5.shape)
pos5[:, :, 0] = X5
pos5[:, :, 1] = Y5

def m_g(pos, mean, sd):
    n = mean.shape[0]
    sd_det = np.linalg.det(sd)
    sd_inv = np.linalg.inv(sd)
    N =np.sqrt(np.power(2 * np.pi, n) * sd_det)
    fac = np.einsum('ijk,kl,ijl->ij', pos-mean, sd_inv, pos-mean)
    return np.exp(-fac / 2) / N

Z1 = m_g(pos5, mean1, sigma1)
Z2 = m_g(pos5, mean2, sigma2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X5, Y5, Z1, rstride=8, cstride=8,  alpha=0.4, cmap=cm.ocean)
#cset = ax.contour(X5, Y5, Z1, zdir='z', offset=-0.18, cmap=cm.ocean)

ax.plot_surface(X5, Y5, Z2, rstride=8, cstride=8, alpha=0.4, cmap=cm.ocean)
#cset = ax.contour(X5, Y5, Z2, zdir='z', offset=-0.17, cmap=cm.ocean)



ax.scatter(class1_valueX,class1_valueY,class1_valueZs,c='r',marker='*');
ax.scatter(class2_valueX,class2_valueY,class2_valueZs,c='b',marker='o');

Z3 = Z1-Z2;
cset = ax.contour(X5, Y5, Z3, zdir='z', offset=-0.17, cmap=cm.ocean)

ax.set_zlabel("Probability Density");
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_zlim(-0.25,0.6)
ax.view_init(30, -125)


plt.xlabel("X")
plt.ylabel("Y")
plt.title("Implementing Minimum Error Rate Classifier.");
plt.show()