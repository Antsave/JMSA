import numpy as np



# 1. Sphere Benchmark (f1) - unimodal
def sphere(x): 
    return np.sum(x**2)

# 2. Absolute Sum + Absolute Product (f2) - unimodal
def asap(x):
    x = np.array(x)
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# 3. Rastrigin Function (f3) - multimodal w many local minima
def rastrigin(x):
    x = np.array(x)
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))

# 4. Griewank Function (f4) - multimodal w many local minima
def griewank(x): 
    x = np.array(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

# 5. Six Hump Camel (f5) - multimodal w few local minima
# def shc(x): 
#     x1 = x[0]
#     x2 = x[1]
#     return (
#         4 * x1**2
#         - 2.1 * x1**4
#         + (x1**6) / 3
#         + x1 * x2
#         - 4 * x2**2
#         + 4 * x2**4
#     )
def shc(x):
    x1, x2 = x[0], x[1]
    return (
        4*x1**2
        - 2.1*x1**4
        + (1/3)*x1**6
        + x1*x2
        - 4*x2**2
        + 4*x2**4
    )


# 6. Branin Function (f6) - multimodal w few local minima
def branin(x):
    x1, x2 = x[0], x[1]
    pi = np.pi

    term1 = x2 - (5.1 / (4 * pi**2)) * x1**2 + (5 / pi) * x1 - 6
    term2 = 10 * (1 - 1 / (8 * pi)) * np.cos(x1)

    return term1**2 + term2 + 10
