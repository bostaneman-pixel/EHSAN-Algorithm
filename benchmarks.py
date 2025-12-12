import numpy as np

def sphere(x):
    return np.sum(x*x)

def ackley(x):
    x = np.array(x)
    a = 20; b = 0.2; c = 2*np.pi
    d = len(x)
    return -a*np.exp(-b*np.sqrt(np.sum(x*x)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.e

def rastrigin(x):
    x = np.array(x)
    A = 10
    return A*len(x) + np.sum(x*x - A*np.cos(2*np.pi*x))

def griewank(x):
    x = np.array(x)
    return np.sum(x*x)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1)))) + 1

def schwefel(x):
    x = np.array(x)
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))
