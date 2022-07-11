import numpy as np
import math
from sympy import Symbol, re
from sympy.solvers import solve
import SLAE

epsilon = 1e-6

a = 1.1
b = 2.5
alph = 0.4
accurate = 18.60294785731848208626949366919856494853

# a = 1.5
# b = 3.3
# alph = 1/3
# accurate = 7.077031437995793610263911711602477164432


def f(x):
    return 0.5 * math.cos(2*x)*math.exp(2*x/5) + 2.4*math.sin(1.5*x)*math.exp(-6*x) + 6*x
    #return 2 * math.cos(2.5 * x) * math.exp(x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + x


#Функция генерации 3 моментов на интервале [z0,z1]
def get_three_moments(z0, z1, a, alpha):
    first = z1 - a
    second = z0 - a
    m0 = (math.pow(first, 1 - alpha) - math.pow(second, 1 - alpha)) / (1 - alpha)
    m1 = (math.pow(first, 2 - alpha) - math.pow(second, 2 - alpha)) / (2 - alpha) + a*m0
    m2 = (math.pow(first, 3 - alpha) - math.pow(second, 3 - alpha)) / (3 - alpha) + 2*a*m1 - a**2*m0
    return np.array([m0, m1, m2])


# генирирует матрицы по узлам xj для решения СЛАУ (x-вектор)
def generate_matrix(X):
    #A = np.ones((X.size, X.size,))
    A = np.ones((X.size, X.size))
    for i in range(1, A.shape[0]):
        A[i] = A[i-1] * X
    return A


# Функция, реализующая метод Ньютона-Котса по 3 точкам
def NK(fun, a, b, alpha, h):
    z0 = z1 = a
    summa = 0
    while not abs(z1 - b) < epsilon:
        z0 = z1
        z1 = z1 + h
        if z1 > b:
            z1 = b
        M = get_three_moments(z0, z1, a, alpha)
        X = np.array([z0, (z0+z1)/2, z1])
        L = generate_matrix(X)

        A, _ = SLAE.SLAE(L, M)
        summa += np.array([i for i in map(fun, X)]) @ A

    return summa


print("\n\nНьютон-Котс =", NK(f, a, b, alph, b-a))
# Правило Рунге + Процесс Эйткена
print("\nСоставная квадратурная формула")
L = 2  # Во сколько раз уменьшается интервал разбиения
lnL = math.log(L)
h0 = b - a
n0 = 1
h1 = h0 / L
n1 = n0 * L
h2 = h1 / L
n2 = n1 * L
Sh0 = NK(f, a, b, alph, h0)
Sh1 = NK(f, a, b, alph, h1)
while True:
    print(h2, n2)

    Sh2 = NK(f, a, b, alph, h2)

    m = - math.log((Sh2 - Sh1) / (Sh1 - Sh0)) / lnL
    print("m =", m)

    Rh2 = (Sh2 - Sh1) / (L ** m - 1)
    print("R =", Rh2)

    if abs(Rh2) < epsilon:
        print("Result:", Sh2 + Rh2)
        print("Accurate:", accurate)
        print(h2)
        break
    h0 = h1
    n0 = n1
    h1 = h2
    n1 = n2
    h2 = h2 / L
    n2 = n2 * L
    Sh0 = Sh1
    Sh1 = Sh2


print("\nНахождение оптимального разбиения")
L = 2
lnL = math.log(L)
h = b - a
n = 1
while True:
    h0 = h
    h1 = h0 / L
    h2 = h1 / L

    Sh0 = NK(f, a, b, alph, h0)
    Sh1 = NK(f, a, b, alph, h1)
    Sh2 = NK(f, a, b, alph, h2)

    m = - math.log(abs((Sh2 - Sh1) / (Sh1 - Sh0))) / lnL
    print("m =", m)

    Rh2 = (Sh2 - Sh1) / (L ** m - 1)
    print("R =", Rh2)

    hopt = 0.95 * h2 * math.pow(epsilon / abs(Rh2), 1 / m)
    print("opt =", hopt)
    print()

    Sh0 = NK(f, a, b, alph, L * L * hopt)
    Sh1 = NK(f, a, b, alph, L * hopt)
    Sh2 = NK(f, a, b, alph, hopt)

    m = - math.log(abs((Sh2 - Sh1) / (Sh1 - Sh0))) / lnL
    print("mopt =", m)

    Rh2 = (Sh2 - Sh1) / (L ** m - 1)
    print("Ropt =", Rh2)

    if abs(Rh2) < epsilon:
        print("Result:", Sh2 + Rh2)
        print("Accurate:", accurate)
        print("Разница:", abs(accurate - Sh2 - Rh2))
        break

    h = hopt


def get_six_moments(z0, z1, a, alpha):
    first = z1 - a
    second = z0 - a
    m0 = (math.pow(first, 1 - alpha) - math.pow(second, 1 - alpha)) / (1 - alpha)
    m1 = (math.pow(first, 2 - alpha) - math.pow(second, 2 - alpha)) / (2 - alpha) + a * m0
    m2 = (math.pow(first, 3 - alpha) - math.pow(second, 3 - alpha)) / (3 - alpha) + 2 * a * m1 - a ** 2 * m0
    m3 = (math.pow(first, 4 - alpha) - math.pow(second, 4 - alpha)) / (
                4 - alpha) + 3 * a * m2 - 3 * a ** 2 * m1 + a ** 3 * m0
    m4 = (math.pow(first, 5 - alpha) - math.pow(second, 5 - alpha)) / (
                5 - alpha) + 4 * a * m3 - 6 * a ** 2 * m2 + 4 * a ** 3 * m1 - a ** 4 * m0
    m5 = (math.pow(first, 6 - alpha) - math.pow(second, 6 - alpha)) / (
                6 - alpha) + 5 * a * m4 - 10 * a ** 2 * m3 + 10 * a ** 3 * m2 - 5 * a ** 4 * m1 + a ** 5 * m0

    return np.array([m0, m1, m2, m3, m4, m5])


def Gauss(fun, a, b, alpha, h, n=3):
    z0 = z1 = a
    summa = 0
    while z1 < b:
        z0 = z1
        z1 = z1 + h
        if z1 > b:
            z1 = b

        M = get_six_moments(z0, z1, a, alpha)
        A = np.zeros((n, n))
        bm = np.zeros(n)

        for i in range(n):
            A[i] = M[i:i + n]
            bm[i] = -M[i + n]

        aa, _ = SLAE.SLAE(A, bm)  # вектор a_j

        # Численное решение уравнения
        x = Symbol('x')
        l = x ** n
        for i in range(n - 1, -1, -1):
            l += aa[i] * x ** i

        s = []
        for x in solve(l, x):
            s.append(re(x))

        L = generate_matrix(np.array(s))

        X, _ = SLAE.SLAE(L, M[:n])

        summa += np.array([i for i in map(fun, s)]) @ X

    return summa


print("\n\nGauss =", Gauss(f, a, b, alph, b-a))

print("\nСоставная квадратурная формула")
L = 2
lnL = math.log(L)
h0 = b - a
h1 = h0 / L
h2 = h1 / L
Sh0 = Gauss(f, a, b, alph, h0)
Sh1 = Gauss(f, a, b, alph, h1)
while True:
    print(h2)

    Sh2 = Gauss(f, a, b, alph, h2)

    m = - math.log((Sh2 - Sh1) / (Sh1 - Sh0)) / lnL
    print("m =", m)

    Rh2 = (Sh2 - Sh1) / (L ** m - 1)
    print("R =", Rh2)

    if abs(Rh2) < epsilon:
        print("Result:", Sh2 + Rh2)
        print("Accurate:", accurate)
        print("Разница:", abs(accurate - Sh2 - Rh2))
        break

    h0 = h1
    h1 = h2
    h2 = h2 / L
    Sh0 = Sh1
    Sh1 = Sh2


print("\nНахождение оптимального разбиения")
L = 2
lnL = math.log(L)
h = b - a
h0 = h
h1 = h0 / L
h2 = h1 / L
while True:
    Sh0 = Gauss(f, a, b, alph, h0)
    Sh1 = Gauss(f, a, b, alph, h1)
    Sh2 = Gauss(f, a, b, alph, h2)

    m = - math.log(abs((Sh2 - Sh1) / (Sh1 - Sh0))) / lnL
    print("m =", m)

    Rh2 = (Sh2 - Sh1) / (L ** m - 1)
    print("R =", Rh2)

    hopt = 0.8 * h2 * math.pow(epsilon / abs(Rh2), 1 / m)
    print("opt =", hopt)

    if hopt > h2:
        print("h2 =", h2)
        print("hopt > h2. Exit")
        break

    Sh0 = Gauss(f, a, b, alph, L * L * hopt)
    Sh1 = Gauss(f, a, b, alph, L * hopt)
    Sh2 = Gauss(f, a, b, alph, hopt)

    m = - math.log(abs((Sh2 - Sh1) / (Sh1 - Sh0))) / lnL
    print("mopt =", m)

    Rh2 = (Sh2 - Sh1) / (L ** m - 1)
    print("Ropt =", Rh2)

    if abs(Rh2) < epsilon:
        print("Result:", Sh2 + Rh2)
        print("Accurate:", accurate)
        break

    h2 = hopt
    h1 = h2 * L
    h0 = h1 * L