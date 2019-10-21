import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt;
from matplotlib import cm
import random
random.seed(0)

df = pd.read_csv('data5.txt',sep=" ",header = None);

arr = df.values;


total_data = len(arr);

wx = [] #x_values
wy = [] #y_values

cx,cy= [],[] #cluster list

for i in range(total_data):
        wx.append(arr[i][0])
        wy.append(arr[i][1])

K =2;

random_pick=[]
for i in range(K):
  r=random.randint(0,total_data-1)
  if r not in random_pick: random_pick.append(r)
print(random_pick)

for i in range(len(random_pick)):
        cx.append(arr[random_pick[i], 0])
        cy.append(arr[random_pick[i], 1])

print(cx)
print(cy)

class1_listX, class1_listY, class2_listX, class2_listY = [], [], [], []

nclass1_listX, nclass1_listY, nclass2_listX, nclass2_listY = [], [], [], []

for i in range(total_data):
        val11 = np.power((arr[i][0] - cx[0]), 2);
        val12 = np.power((arr[i][1] - cy[0]), 2);
        distance1 = np.sqrt(val11 + val12)

        val21 = np.power((arr[i][0] - cx[1]), 2);
        val22 = np.power((arr[i][1] - cy[1]), 2);
        distance2 = np.sqrt(val21 + val22)
        # print("Distance for ",test[i],": is ",val3)
        if distance1<=distance2:
                class1_listX.append(arr[i][0])
                class1_listY.append(arr[i][1])
        elif distance1>distance2:
                class2_listX.append(arr[i][0])
                class2_listY.append(arr[i][1])



cx1,cy1 = sum(class1_listX)/len(class1_listX), sum(class1_listY)/len(class1_listY);
cx2,cy2 = sum(class2_listX)/len(class2_listX), sum(class2_listY)/len(class2_listY);

x=0;
while x < 100:
        for i in range(total_data):
                val11 = np.power((arr[i][0] - cx1), 2);
                val12 = np.power((arr[i][1] - cy1), 2);
                distance1 = np.sqrt(val11 + val12)

                val21 = np.power((arr[i][0] - cx2), 2);
                val22 = np.power((arr[i][1] - cy2), 2);
                distance2 = np.sqrt(val21 + val22)
                # print("Distance for ",test[i],": is ",val3)
                if distance1 <= distance2:
                        nclass1_listX.append(arr[i][0])
                        nclass1_listY.append(arr[i][1])
                elif distance1 > distance2:
                        nclass2_listX.append(arr[i][0])
                        nclass2_listY.append(arr[i][1])
        if class1_listX==nclass1_listX and class1_listY==nclass1_listY and class2_listX==nclass2_listX and class2_listY==nclass2_listY:
                print("Yes")
                break;
        else:
             cx1, cy1 = sum(nclass1_listX) / len(nclass1_listX), sum(nclass1_listY) / len(nclass1_listY);
             cx2, cy2 = sum(nclass2_listX) / len(nclass2_listX), sum(nclass2_listY) / len(nclass2_listY);
             class1_listX = nclass1_listX;
             class1_listY = nclass1_listY;
             class2_listX = nclass2_listX;
             class2_listY = nclass2_listY;

        x = x + 1;

print((x+1))
print(nclass1_listX)
print(nclass1_listY)

#print(total_class1,total_class2)


plt.plot(arr[:,0],arr[:,1],'*b',label='Without K-Means Clustering');
plt.title('Implementing K-Means Clustering(Before)')
plt.legend();
plt.show();


plt.plot(nclass1_listX,nclass1_listY,'og',label='class1 Data');
plt.plot(nclass2_listX,nclass2_listY,'or',label='class2 Data');

plt.title('Implementing K-Means Clustering(After)')
plt.legend();
plt.show();
