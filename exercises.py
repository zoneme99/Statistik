import matplotlib.pyplot as plt
import numpy as np

x1 = np.array(([0,2,0,0,-1,1]))
x2 = np.array([3,0,1,1,0,1])
x3 = np.array([0,0,3,2,1,1])
y = ["r","r","r","g","g","r"]

test = np.array((2,2,2))

for index in range(len(x1)):
    current = np.linalg.norm(test - (x1[index],x2[index],x3[index]))
    if index == 0:
        dist = current
        position = index
        continue
    
    if dist > current:
        dist = current
        position = index

color_map = {"r":"red", "g":"green"}

print(f"closest neighbor is {color_map[y[position]]}")

matrix = np.reshape(np.concat([x1,x2,x3]),(3,6)).T
print(matrix)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x1,x2,x3, c=y)
ax.scatter3D(test[0],test[1],test[2], c="black")

plt.show()