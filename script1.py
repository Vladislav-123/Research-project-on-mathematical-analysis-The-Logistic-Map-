import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    return r * x * (1 - x)

x = np.linspace(0, 1, 500)

for r in [0.5, 1, 2, 3, 4]:
    plt.plot(x, logistic(x, r), label=f"r = {r}")

plt.xlabel(r"$x_{n-1}$")
plt.ylabel(r"$x_n$")
plt.title('Логистическое отображение')
plt.grid()
plt.legend()
plt.show()


def g(x, r):
    return r * x * (1 - x)**2

x = np.linspace(0, 1, 500)

for r in [1, 3, 5, 6.75]:
    plt.plot(x, g(x, r), label=f"r = {r}")

plt.xlabel(r"$x_{n-1}$")
plt.ylabel(r"$x_n$")
plt.title(r'Отображение $g(x)=rx(1-x)^2$')
plt.grid()
plt.legend()
plt.show()


r = 3
x = np.linspace(0, 1, 500)

plt.plot(x, logistic(x, r), label=r"$rx(1-x)$")
plt.plot(x, g(x, r), label=r"$rx(1-x)^2$")

plt.xlabel(r"$x_{n-1}$")
plt.ylabel(r"$x_n$")
plt.title(f'Сравнение отображений при r = {r}')
plt.grid()
plt.legend()
plt.show()
