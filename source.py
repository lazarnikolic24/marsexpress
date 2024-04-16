import numpy as np
import matplotlib.pyplot as plt

t,x,y = np.loadtxt("marsexpresslr.d", usecols = (0,1,2), unpack = True)
n = len(t)
dt = t[1] - t[0]
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
AU = 129598000.0
r = r/AU

plt.plot(r[:, 0], r[:, 1])
plt.axis("equal")
plt.xlabel("x [au]")
plt.ylabel("y [au]")
#plt.show()

# motion diagram

for i in range(n - 1):
    plt.plot(r[i, 0], r[i, 1], "o")
    dr = r[i + 1, :] - r[i, :]
    plt.quiver(r[i, 0], r[i, 1], dr[0], dr[1], angles = "xy",
        scale_units = "xy", scale = 1)

for i in range(1, n - 1):
    plt.plot(r[i, 0], r[i, 1], "o")
    dr = (r[i + 1, :] - r[i, :]) - (r[i, :] - r[i - 1, :])
    plt.quiver(r[i, 0], r[i, 1], dr[0], dr[1], angles = "xy",
        scale_units = "xy", scale = 1)

# висока резолуција податци

t,x,y = np.loadtxt("marsexpresshr.d", usecols = (0,1,2), unpack = True)
n = len(t)
dt = t[1] - t[0]
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
r = r/AU

n = len(r)
v = np.zeros((n, 2), float)
for i in range(n - 1):
    v[i, :] = (r[i+1,:]-r[i,:]) / dt

a = np.zeros((n, 2), float)
for i in range(2, n - 1):
    a[i, :] = (v[i, :] - v[i - 1, :]) / dt

vv = np.zeros((n,1),float)
aa = np.zeros((n,1),float)
for i in range(n):
    vv[i] = np.linalg.norm(v[i,:])
    aa[i] = np.linalg.norm(a[i,:])



plt.show()
