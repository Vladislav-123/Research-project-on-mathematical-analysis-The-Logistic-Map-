import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    return r * x * (1 - x)

def find_period(f, r, x0=0.2, transient=5000,
                check=500, max_period=128, tol=1e-8):
    x = x0

    # Отбрасываем переходный процесс
    for _ in range(transient):
        x = f(x, r)

    # Сохраняем установившуюся часть траектории
    values = []
    for _ in range(check + max_period):
        x = f(x, r)
        values.append(x)

    values = np.array(values)

    # Ищем минимальный период
    for p in range(1, max_period + 1):
        if np.allclose(values[p:], values[:-p], atol=tol, rtol=0):
            return p

    return None

for r in [3.2, 3.5, 3.55, 3.565]:
    print(r, find_period(logistic, r))


def get_trajectory(f, x0, r, transient=1000, steps=40):
    x = x0

    for _ in range(transient):
        x = f(x, r)

    values = []
    for _ in range(steps):
        x = f(x, r)
        values.append(x)

    return np.array(values)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

r_values = [3.2, 3.5, 3.55, 3.565]
axes_flat = axes.flat

for i in range(len(r_values)):
    r = r_values[i]
    ax = axes_flat[i]

    values = get_trajectory(logistic, 0.2, r)

    ax.plot(values, 'o-', markersize=3)
    ax.set_title(f'r = {r}')
    ax.set_xlabel('n')
    ax.set_ylabel(r'$x_n$')
    ax.grid()

plt.tight_layout()
plt.show()


def cobweb(f, r, x0, steps=30, title='Лестница Ламерея'):
    x = np.linspace(0, 1, 1000)

    plt.figure(figsize=(7, 7))
    plt.plot(x, f(x, r), label=r'$f(x)$')
    plt.plot(x, x, 'k--', label=r'$y=x$')

    xn = x0
    yn = f(xn, r)

    # Первый шаг из (x0, 0)
    plt.plot([xn, xn], [0, yn], color='red')

    for _ in range(steps):
        # Из (xn, x_{n+1}) горизонтально к диагонали
        plt.plot([xn, yn], [yn, yn], color='red')

        xn = yn
        yn = f(xn, r)

        # Вертикально к графику отображения
        plt.plot([xn, xn], [xn, yn], color='red')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'$x_n$')
    plt.ylabel(r'$x_{n+1}$')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()

params = [
    (3.2, 20),
    (3.5, 40),
    (3.55, 60)
]
for r, steps in params:
    cobweb(
        logistic,
        r=r,
        x0=0.2,
        steps=steps,
        title=f'Лестница Ламерея: r = {r}'
    )


def g(x, r):
    return r * x * (1 - x)**2

for r in [3.5, 4.2, 5.0, 5.5, 6.0, 6.5]:
    period = find_period(
        g, r,
        x0=0.2,
        transient=10000,
        check=1000,
        max_period=256,
        tol=1e-8
    )
    print(r, period)


r_values = np.linspace(4.01, 6.74, 700)

periods = []

for r in r_values:
    p = find_period(
        g, r,
        x0=0.2,
        transient=5000,
        check=500,
        max_period=128,
        tol=1e-7
    )
    periods.append(p if p is not None else np.nan)

plt.figure(figsize=(9, 5))
plt.scatter(r_values, periods, s=5)

plt.yscale('log', base=2)
plt.xlabel('r')
plt.ylabel('Период m')
plt.title(r'Изменение периода цикла для $g(x)=rx(1-x)^2$')
plt.grid()
plt.show()


r_values = np.linspace(0.01, 6.75, 3000)

rs = []
xs = []

for r in r_values:
    x = 0.2

    for _ in range(2000):
        x = g(x, r)

    for _ in range(100):
        x = g(x, r)
        rs.append(r)
        xs.append(x)

plt.figure(figsize=(10, 6))
plt.plot(rs, xs, ',k', alpha=0.5)

plt.xlabel('r')
plt.ylabel(r'$x_n$')
plt.title(r'Установившиеся значения для $g(x)=rx(1-x)^2$')
plt.xlim(0, 6.75)
plt.ylim(0, 1)
plt.show()


for r in [3.5, 4.2, 5.0]:
    cobweb(
        g,
        r=r,
        x0=0.2,
        steps=60,
        title=rf'$g(x)=rx(1-x)^2$, r = {r}'
    )
