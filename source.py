import numpy as np
import matplotlib.pyplot as plt

t,x,y = np.loadtxt("marsexpresslr.d", usecols = (0,1,2), unpack = True)
n = len(t)
dt = t[1] - t[0]
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
AU = 149598000.0
r = r/AU

fig1 = plt.figure()

plt.subplot(1, 2, 1, title = "lowres data")
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

plt.subplot(1, 2, 2, aspect = "equal", title = "highres data")

t,x,y = np.loadtxt("marsexpresshr.d", usecols = (0,1,2), unpack = True)
n = len(t)
dt = t[1] - t[0]
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
r = r/AU
plt.plot(r[:, 0], r[:, 1])
plt.axis("equal")
plt.xlabel("x [au]")
plt.ylabel("y [au]")

# zemlja i marss

RE = 1.0
TE = 365.25
tt = np.linspace(0, TE, 1000)
omegaE = 2 * np.pi / TE
rE = RE * np.transpose(np.array([np.cos(omegaE * tt), np.sin(omegaE*tt)]))
plt.plot(rE[:, 0], rE[:, 1], ":", color = "g")
RM = 1.5
TM = 2 * 365.25
tt = np.linspace(0, TM, 1000)
omegaM = 2 * np.pi / TM
rM = RM * np.transpose(np.array([np.cos(omegaM*tt), np.sin(omegaM*tt)]))
plt.plot(rM[:, 0], rM[:, 1], "--", color = "r")

# brzina i ubrzanje

n = len(r)
v = np.zeros((n, 2), float)
for i in range(n - 1):
    v[i, :] = (r[i + 1, :] - r[i, :]) / dt

a = np.zeros((n, 2), float)
for i in range(2, n - 1):
    a[i, :] = (v[i, :] - v[i - 1, :]) / dt

vv = np.zeros((n,1),float)
aa = np.zeros((n,1),float)
for i in range(n):
    vv[i] = np.linalg.norm(v[i,:])
    aa[i] = np.linalg.norm(a[i,:])

fig2 = plt.figure()

plt.subplot(3, 1, 1, title = "velocity")
plt.plot(t, vv)
plt.axis("equal")
plt.xlabel("t [days]")
plt.ylabel("v [au/days]")


plt.subplot(3, 1, 2, title = "acceleration")
plt.plot(t, aa)
plt.axis("equal")
plt.xlabel("t [days]")
plt.ylabel("a [au/days^2]")

rr = np.zeros((n,1),float)
for i in range(n):
    rr[i] = np.linalg.norm(r[i,:])

plt.subplot(3, 1, 3, title = "acceleration")
plt.plot(1/rr**2, aa)
plt.axis("equal")
plt.xlabel("t [days]")
plt.ylabel("v [au/days]")



plt.show()