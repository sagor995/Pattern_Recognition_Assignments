#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt;
from matplotlib import cm
import random
random.seed(123)

df = pd.read_csv('irisdataset.txt',sep="\t",header = None);

arr = df.values;

print(arr)


total_data = len(arr);

wx = [] #x_values
wy = [] #y_values
wz = [] #z_values

cx,cy,cz= [],[],[] #cluster list

for i in range(total_data):
        wx.append(arr[i][0])
        wy.append(arr[i][1])
        wz.append(arr[i][2])

K =3;

random_pick=[]
for i in range(K):
  r=random.randint(0,total_data-1)
  if r not in random_pick: random_pick.append(r)
print(random_pick)

for i in range(len(random_pick)):
        cx.append(arr[random_pick[i], 0])
        cy.append(arr[random_pick[i], 1])
        cz.append(arr[random_pick[i], 2])

print(cx)
print(cy)
print(cz)

class1_listX, class1_listY,class1_listZ, class2_listX, class2_listY,class2_listZ, class3_listX, class3_listY,class3_listZ = [], [], [], [],[],[],[],[],[]

nclass1_listX, nclass1_listY,nclass1_listZ, nclass2_listX, nclass2_listY, nclass2_listZ, nclass3_listX, nclass3_listY, nclass3_listZ = [], [], [], [],[],[],[],[],[]


for i in range(total_data):
        val11 = np.power((arr[i][0] - cx[0]), 2);
        val12 = np.power((arr[i][1] - cy[0]), 2);
        val13 = np.power((arr[i][2] - cz[0]), 2);

        distance1 = np.sqrt(val11 + val12 + val13)

        val21 = np.power((arr[i][0] - cx[1]), 2);
        val22 = np.power((arr[i][1] - cy[1]), 2);
        val23 = np.power((arr[i][1] - cz[1]), 2);
        distance2 = np.sqrt(val21 + val22+val23)

        val31 = np.power((arr[i][0] - cx[2]), 2);
        val32 = np.power((arr[i][1] - cy[2]), 2);
        val33 = np.power((arr[i][1] - cz[2]), 2);
        distance3 = np.sqrt(val31 + val32 + val33)

        dis_min = min(distance1,distance2,distance3);


        # print("Distance for ",test[i],": is ",val3)
        if dis_min==distance1:
                class1_listX.append(arr[i][0])
                class1_listY.append(arr[i][1])
                class1_listZ.append(arr[i][2])
        elif dis_min==distance2:
                class2_listX.append(arr[i][0])
                class2_listY.append(arr[i][1])
                class2_listZ.append(arr[i][2])
        elif dis_min==distance3:
                class3_listX.append(arr[i][0])
                class3_listY.append(arr[i][1])
                class3_listZ.append(arr[i][2])


print(class2_listX)
print(class2_listY)
print(class2_listZ)

cx1,cy1,cz1 = sum(class1_listX)/len(class1_listX), sum(class1_listY)/len(class1_listY), sum(class1_listZ)/len(class1_listZ);
cx2,cy2,cz2 = sum(class2_listX)/len(class2_listX), sum(class2_listY)/len(class2_listY), sum(class2_listZ)/len(class2_listZ);
cx3,cy3,cz3 = sum(class3_listX)/len(class3_listX), sum(class3_listY)/len(class3_listY), sum(class3_listZ)/len(class3_listZ);


x=0;
while x < 100:
        for i in range(total_data):
            val11 = np.power((arr[i][0] - cx1), 2);
            val12 = np.power((arr[i][1] - cy1), 2);
            val13 = np.power((arr[i][2] - cz1), 2);

            distance1 = np.sqrt(val11 + val12 + val13)

            val21 = np.power((arr[i][0] - cx2), 2);
            val22 = np.power((arr[i][1] - cy2), 2);
            val23 = np.power((arr[i][1] - cz2), 2);
            distance2 = np.sqrt(val21 + val22 + val23)

            val31 = np.power((arr[i][0] - cx3), 2);
            val32 = np.power((arr[i][1] - cy3), 2);
            val33 = np.power((arr[i][1] - cz3), 2);
            distance3 = np.sqrt(val31 + val32 + val33)

            dis_min = min(distance1, distance2, distance3);

            # print("Distance for ",test[i],": is ",val3)
            if dis_min == distance1:
                nclass1_listX.append(arr[i][0])
                nclass1_listY.append(arr[i][1])
                nclass1_listZ.append(arr[i][2])
            elif dis_min == distance2:
                nclass2_listX.append(arr[i][0])
                nclass2_listY.append(arr[i][1])
                nclass2_listZ.append(arr[i][2])
            elif dis_min == distance3:
                nclass3_listX.append(arr[i][0])
                nclass3_listY.append(arr[i][1])
                nclass3_listZ.append(arr[i][2])
        if class1_listX==nclass1_listX and class1_listY==nclass1_listY and class1_listZ==nclass1_listZ and class2_listX==nclass2_listX and class2_listY==nclass2_listY and class2_listZ==nclass2_listZ and class3_listX==nclass3_listX and class3_listY==nclass3_listY and class3_listZ==nclass3_listZ:
                print("Yes")
                break;
        else:
            cx1, cy1, cz1 = sum(class1_listX) / len(class1_listX), sum(class1_listY) / len(class1_listY), sum(
                class1_listZ) / len(class1_listZ);
            cx2, cy2, cz2 = sum(class2_listX) / len(class2_listX), sum(class2_listY) / len(class2_listY), sum(
                class2_listZ) / len(class2_listZ);
            cx3, cy3, cz3 = sum(class3_listX) / len(class3_listX), sum(class3_listY) / len(class3_listY), sum(
                class3_listZ) / len(class3_listZ);
            class1_listX = nclass1_listX;
            class1_listY = nclass1_listY;
            class1_listZ = nclass1_listZ;
            class2_listX = nclass2_listX;
            class2_listY = nclass2_listY;
            class2_listZ = nclass2_listZ;
            class3_listX = nclass3_listX;
            class3_listY = nclass3_listY;
            class3_listZ = nclass3_listZ;

        x = x + 1;



print((x+1))
print(nclass1_listX)
print(nclass1_listY)
print(nclass1_listZ)

print("Centroid 1:",cx1,cy1,cz1)
print("Centroid 2:",cx2,cy2,cz2)
print("Centroid 3:",cx3,cy2,cz3)
print(nclass1_listY)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(nclass1_listX,nclass1_listY,nclass1_listZ,s=100,c='r',marker='*',label='Centroid1');
ax.scatter(nclass2_listX,nclass2_listY,nclass2_listZ,s=100,c='g',marker='o',label='Centroid2');
ax.scatter(nclass3_listX,nclass3_listY,nclass3_listZ,s=100,c='b',marker='^',label='Centroid3');
plt.title('Implementing K-Means Clustering(3D)')
plt.legend();
plt.show()
