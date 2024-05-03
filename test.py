import numpy as np
import matplotlib.pyplot as plt

#loading low-res data
t, x, y = np.loadtxt("marsexpresslr.d",  usecols = (0, 1, 2), unpack = True)
n = len(t)
dt = t[1] - t[0] #jer su podaci u fajlu za periode u razmaku od 30 dana

r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
AU = 149598000.0
r = r/AU

fig1 = plt.figure()

#ploting low-res data
plt.subplot(1, 2, 1, title = "low-res data")
plt.plot(r[:, 0], r[:, 1], "o")
plt.axis("equal")
plt.xlabel("x [au]")
plt.ylabel("y [au]")

#ploting average velocity and acceleration using lowres data
for i in range(n - 1):
    #plt.plot(r[i, 0], r[i, 1], "o")
    dr = r[i+1, :] - r[i, :]
    plt.quiver(r[i, 0], r[i, 1], dr[0], dr[1],
                angles = "xy", scale_units = "xy", scale = 1)
    
for i in range(1, n-1):
    #plt.plot(r[i, 0],  r[i, 1], "o")
    dr = (r[i+1, :] - r[i, :]) - (r[i, :] - r[i-1, :])
    plt.quiver(r[i, 0], r[i, 1], dr[0], dr[1],
            angles = "xy", scale_units = "xy", scale = 1)

#high-res data
t, x, y = np.loadtxt("marsexpresshr.d", usecols = (0, 1, 2), unpack = True)
n = len(t)

r = np.zeros((n, 2), float)
r[:, 0] = x
r[:, 1] = y
r = r/AU

plt.subplot(1, 2, 2, title = "high-res data")
plt.plot(r[:, 0], r[:, 1])
plt.axis("equal")
plt.xlabel("x [au]")
plt.ylabel("y [au]")

#putanja zemlje i marsa

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

#intenzitet brzine i ubrzanja zemlje i marsa su konstantne
VE = 2*np.pi * RE/TE;
VM = 2*np.pi * RM/TM;

AE = 4 * np.pi**2 * RE / TE**2
AM = 4* np.pi**2 * RM / TM**2

#racunanje brzine i ubrzanja high-res data
n = len(r)
dt = t[1] - t[0]

v = np.zeros((n-1, 2), float)
for i in range (n-1):
    v[i, :] = (r[i+1, :] - r[i, :]) / dt;

vv = np.zeros((n-1, 1), float)
for i in range (n-1):
    vv[i] = np.linalg.norm(v[i, :])

a = np.zeros((n-1, 2), float)
for i in range (1, n-1):
    a[i, :] = (v[i, :] - v[i-1, :]) / dt;

aa = np.zeros((n-1, 1), float)
for i in range (1, n-1):
    aa[i] = np.linalg.norm(a[i, :])


plt.subplots_adjust(wspace=0.3)

#plotovanje brzine i ubrzanja high-res data
fig2 = plt.figure()

plt.subplot(3, 1, 1, title = "high-res velocity")
plt.plot(t[0 : n-1], vv[0 : n-1])
plt.xlabel("t[days]")
plt.ylabel("v[au/days]")
plt.plot(t[0 : n], np.full((n), VE, dtype=float), ":")
plt.plot(t[0 : n], np.full((n), VM, dtype=float), "--")

#! n-5 za krajnju velicinu je hardkodovanje, mozda postoji greska negde u racunanju
#! takodje u knjizi grafik ubrzanja ne prelazi donju isprekidanu liniju
plt.subplot(3, 1, 2, title = "high-res acceleration")
plt.plot(t[1 : n-5], aa[1 : n-5])
plt.xlabel("t[days]")
plt.ylabel("a[au/days^2]")
plt.yscale('log')
#plt.ylim(0.0001, 0.0004)
plt.plot(t[1 : n-5], np.full((n-6), AE, dtype=float), ":")
plt.plot(t[1 : n-5], np.full((n-6), AM, dtype=float), "--")



#!plotovanje koje pokazuje obrnutu proporcionalnost ubrzanja od rastojanja^2
#!ne kapiram kako da plotujem ovo cudo, nzm sta treba da bude x-osa, sta y-osa

#last plot
rr = np.zeros((n-1, 1), float)
for i in range(n-1):
    rr[i] = np.linalg.norm(r[i, :])

plt.subplot(3, 1, 3, title="Newton’s law of gravitation")
plt.plot(1/(rr**2), aa)
#plt.plot(np.log(1/(rr**2)), np.log(aa))
#plt.plot(rr, 1/(rr**2))
#plt.plot(rr[0:n-1], aa)
plt.xlabel("1/r^2")
plt.ylabel("a")
#plt.axis([0.4, 1, 0.4, 1])
#plt.xscale('log')
#plt.xlim(1, 100)
#plt.xlim(0.004, 0.1)
#plt.ylim(0.0002, 0.0004)
#plt.xlim(0.4, 1)
#plt.axis('equal')

plt.subplots_adjust(hspace=1, wspace=0.5)

plt.show()


