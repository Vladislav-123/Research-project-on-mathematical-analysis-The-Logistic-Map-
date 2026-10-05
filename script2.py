import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    return r * x * (1 - x)

def trajectory(f, x0, r, steps=30):
    xs = [x0]
    for _ in range(steps):
        xs.append(f(xs[-1], r))
    return np.array(xs)

x0 = 0.8
n = np.arange(31)

for r in [0.3, 0.6, 0.9, 1.0]:
    xs = trajectory(logistic, x0, r, 30)
    plt.plot(n, xs, marker='o', markersize=3, label=f'r = {r}')

plt.xlabel('n')
plt.ylabel(r'$x_n$')
plt.title(r'Сходимость $x_n$ при $r \in (0,1]$')
plt.grid()
plt.legend()
plt.show()


r = 2.8
x_star = 1 - 1 / r
x0 = 0.8

xs = trajectory(logistic, x0, r, 30)
n = np.arange(len(xs))

plt.plot(n[::2], xs[::2], 'o-', label=r'$x_{2n}$')
plt.plot(n[1::2], xs[1::2], 'o-', label=r'$x_{2n+1}$')
plt.axhline(x_star, color='black', linestyle='--', label=r'$x^*$')

plt.xlabel('n')
plt.ylabel(r'$x_n$')
plt.title(f'Подпоследовательности при r = {r}')
plt.grid()
plt.legend()
plt.show()


def g(x, r):
    return r * x * (1 - x)**2

x0 = 0.2
steps = 40
n = np.arange(steps + 1)

for r in [0.5, 1.0, 2.0, 4.0, 6.0]:
    xs = trajectory(g, x0, r, steps)
    plt.plot(n, xs, marker='o', markersize=3, label=f'r = {r}')

plt.xlabel('n')
plt.ylabel(r'$x_n$')
plt.title(r'Траектории для $g(x)=rx(1-x)^2$')
plt.grid()
plt.legend()
plt.show()
