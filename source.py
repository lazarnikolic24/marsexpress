import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sf


#ucitavanje podataka
t,x,y = np.loadtxt("marsexpresslr.d", usecols = (0,1,2), unpack = True)
n = len(t)
dt = t[1] - t[0]
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
AU = 149598000.0
r = r/AU

fig1 = plt.figure()


#low-res podaci
plt.subplots_adjust(wspace=0.3)
plt.subplot(1, 2, 1, title = "lowres data")
plt.scatter(r[:, 0], r[:, 1])
plt.axis("equal")
plt.xlabel("x [au]")
plt.ylabel("y [au]")

for i in range(n - 1):
    #plt.plot(r[i, 0], r[i, 1], "o")
    dr = r[i + 1, :] - r[i, :]
    plt.quiver(r[i, 0], r[i, 1], dr[0], dr[1], angles = "xy",
        scale_units = "xy", scale = 1)

for i in range(1, n - 1):
    #plt.plot(r[i, 0], r[i, 1], "o")
    dr = (r[i + 1, :] - r[i, :]) - (r[i, :] - r[i - 1, :])
    plt.quiver(r[i, 0], r[i, 1], dr[0], dr[1], angles = "xy",
        scale_units = "xy", scale = 1, color = "green")

# high-res podaci
plt.subplot(1, 2, 2, aspect = "equal", title = "highres data")

t,x,y = np.loadtxt("marsexpresshr.d", usecols = (0,1,2), unpack = True)
n = len(t)
r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
r = r/AU
plt.plot(r[:, 0], r[:, 1])
plt.axis("equal")
plt.xlabel("x [au]")
plt.ylabel("y [au]")

# zemlja i mars putanja
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

#intenzitet brzine i ubrzanja zemlje i marsa
VE = 2*np.pi * RE/TE
VM = 2*np.pi * RM/TM

AE = 4 * np.pi**2 * RE / TE**2
AM = 4* np.pi**2 * RM / TM**2

# brzina high-res
n = len(r)
dt = t[1] - t[0]

v = np.zeros((n - 1, 2), float)
for i in range(n - 1):
    v[i, :] = (r[i + 1, :] - r[i, :]) / dt

vv = np.zeros((n - 1, 1), float)
for i in range(n - 1):
    vv[i] = np.linalg.norm(v[i,:])

#plotovanje brzine high-res data
fig2 = plt.figure()

plt.subplot(3, 1, 1, title = "velocity")
plt.plot(t[: n - 1], vv[: n - 1])
plt.xlabel("t [days]")
plt.ylabel("v [au/days]")
plt.plot(t[:], np.full((n), VE, dtype=float), ":")
plt.plot(t[:], np.full((n), VM, dtype=float), "--")

#ubrzanje high-res
a = np.zeros((n - 1, 2), float)
for i in range(1, n - 1):
    a[i, :] = (v[i, :] - v[i - 1, :]) / dt


aa = np.zeros((n - 1, 1), float)
for i in range(1, n - 1):
    aa[i] = np.linalg.norm(a[i, :])


#plotovanje ubrzanja high-res data
plt.subplots_adjust(wspace=0.3)
plt.subplot(3, 1, 2, title = "acceleration")
plt.plot(t[1 : n - 5], aa[1 : n - 5], linewidth = 0.2)
plt.xlabel("t [days]")
plt.ylabel("a [au/days^2]")
plt.plot(t[:], np.full((n), AE, dtype=float), ":")
plt.plot(t[:], np.full((n), AM, dtype=float), "--")

#plotovanje Njutnovog zakona gravitacije
plt.subplot(3, 1, 3, title = "Newton's law of gravitation")

rr = np.zeros((n, 1),float)
for i in range(n):
    rr[i] = np.linalg.norm(r[i, :])

invr2 = 1 / np.square(rr)

smoothed_aa = sf(aa.flatten(), window_length=101, polyorder=3)
smoothed_aa_flat = smoothed_aa.flatten()
invr2_flat = invr2.flatten()

initial_discard = 60
plt.plot(invr2_flat[initial_discard:n-1], smoothed_aa_flat[initial_discard:n-1])

plt.xlabel("1/r^2")
plt.ylabel("a")

plt.subplots_adjust(hspace=1, wspace=0.5)

plt.show()
