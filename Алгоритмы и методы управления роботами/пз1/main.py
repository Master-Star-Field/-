import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.margins(0,0)
t = np.linspace(0, 6*np.pi, 100)


x = np.cos(t)
y = np.sin(t)
z = t

fig = plt.figure()
ax = plot(111, projection='3d')
ax.plot(x, y, z)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



dx_dt = -np.sin(t)
dy_dt = np.cos(t)
dz_dt = np.ones_like(t)

v = np.array([dx_dt, dy_dt, dz_dt])
a = np.array([-np.sin(t), -np.cos(t), np.zeros_like(t)])


fig, axs = plt.subplots(3, 1, figsize=(8, 12))
plt.rcParams['axes.edgecolor'] = 'blue'
plt.rcParams['axes.linewidth'] = 2


axs[0].plot(t, np.sqrt(1+t**2), label='r(t)')
axs[0].set_xlabel('t')
axs[0].set_ylabel('x, y, z')
axs[0].legend()


axs[1].plot(t, np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2), label='V')
axs[1].set_xlabel('t')
axs[1].set_ylabel('dx/dt, dy/dt, dz/dt')
axs[1].legend()


axs[2].plot(t, [1]*100, label='W')
axs[2].set_xlabel('t')
axs[2].set_ylabel('W')
axs[2].legend()

plt.tight_layout()

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# График радиус-вектора
axs[0].plot(t, x, label='x(t)')
axs[0].plot(t, y, label='y(t)')
axs[0].plot(t, z, label='z(t)')
axs[0].set_xlabel('t')
axs[0].set_ylabel('x, y, z')
axs[0].legend()

# График скорости
axs[1].plot(t, dx_dt, label='dx/dt')
axs[1].plot(t, dy_dt, label='dy/dt')
axs[1].plot(t, dz_dt, label='dz/dt')
axs[1].set_xlabel('t')
axs[1].set_ylabel('dx/dt, dy/dt, dz/dt')
axs[1].legend()


# График ускорения
axs[2].plot(t, a[0], label='d^2x/dt^2')
axs[2].plot(t, a[1], label='d^2y/dt^2')
axs[2].plot(t, a[2], label='d^2z/dt^2')
axs[2].set_xlabel('t')
axs[2].set_ylabel('d^2x/dt^2, d^2y/dt^2, d^2z/dt^2')
axs[2].legend()

plt.subplots_adjust(left=0, right=0.9, top=0.9, bottom=0,wspace=0, hspace=0)
plt.tight_layout()

#plt.savefig('image.png', bbox_inches='tight',pad_inches = 0)

plt.show()



