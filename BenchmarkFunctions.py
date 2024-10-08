import numpy as np


# Define benchmark functions
def ackley(X):
    x = X[0]
    y = X[1]
    part1 = -0.20 * np.sqrt(0.5 * (x ** 2 + y ** 2))
    part2 = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return -20.0 * np.exp(part1) - np.exp(part2) + 20 + np.e


def sphere(X):
    return sum([x ** 2 for x in X])


def schwefel_1_2(X):
    total = 0
    for i in range(len(X)):
        sum_j = sum(X[:i+1])
        total += sum_j ** 2
    return total


def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])


def rosenbrock(X):
    total = 0
    for i in range(len(X) - 1):
        xi = X[i]
        xi_next = X[i + 1]
        total += 100 * (xi_next - xi ** 2) ** 2 + (xi - 1) ** 2
    return total


def griewank(X):
    sum_sq = sum([x ** 2 for x in X]) / 4000
    prod_cos = np.prod([np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(X)])
    return sum_sq - prod_cos + 1


def weierstrass(X, a=0.5, b=3, k_max=20):
    n = len(X)
    sum1 = 0
    for xi in X:
        sum1 += sum([a ** k * np.cos(2 * np.pi * b ** k * (xi + 0.5)) for k in range(k_max + 1)])
    sum2 = n * sum([a ** k * np.cos(2 * np.pi * b ** k * 0.5) for k in range(k_max + 1)])
    return sum1 - sum2
