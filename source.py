import numpy as np
import matplotlib.pyplot as plt

t,x,y = np.loadtxt('marsexpresslr.d', usecols = (0,1,2), unpack = True)
n = len(t)
dt = t[1] - t[0]
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
AU = 129598000.0
r = r/AU

plt.plot(r[:, 0], r[:, 1])
plt.axis('equal')
plt.xlabel('x [au]')
plt.ylabel('y [au]')
plt.show()

