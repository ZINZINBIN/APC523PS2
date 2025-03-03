"""
    Problem 2. Integration
        Write a numerical integration function that takes the following arguments: a function f(x), 
        a number of discrete intervals N, and the integration bounds aand b. The function must return 
        the value of integral f(x) from a to b. The integrator should use 5-point Gaussian Quadrature 
        method in each of the N intervals.
        
        Plot the relative error of integration as a function of h= (b−a)/N for
        (a) a=−1,b= 1,f(x) = x8
        (b) a=−1,b= 1,f(x) = |x−1/√2|^3
        (c) a=−1,b= 1,f(x) = H[x−1/√2]
        (d) a= 0,b= 1,f(x) = 1/√x.
        
        Determine the power law dependency of each error curve as a function of h(e.g. h^0.5, h^1, or h^2).
        Then, plot the corresponding power-law trend as a dashed line for reference. Write the explanations
        of the power law dependencies that you find for each function in the write-up.
        
        (Optional) What do you think will happen if we split the integration domain for function (c) 
        into [−1,1/√2] and [1/√2,1]? Can you explain the error scaling that you obtain by splitting
        the domain?
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from numpy.polynomial.legendre import leggauss

# 5-point Gauss quadrature
def generate_Gauss_quadrature(p:int):
    (xg, wg) = leggauss(p)
    return xg, wg

# generate polynomial function with chebyshev nodes
def generate_langrange_polynomial(x:np.array, y:np.array):
    @np.vectorize
    def interp(xn:np.array):
        dn = xn - x
        r = 0
        for i in range(len(x)):
            d = x[i] - x
            L = np.prod(dn[0:i] / d[0:i]) * np.prod(dn[i+1:] / d[i+1:])
            r = r + L * y[i]
        return r
    return interp

def compute_relative_err(pred:float,exact:float):
    err = abs(pred - exact) / (abs(exact) + 1e-8)
    return err

# Integration algorithm using Gaussian quadrature
def compute_integral(a:float, b:float, N:int, p:int, func:Callable):
    h = (b-a) / N
    xns = np.linspace(a,b,N,endpoint = True)
    F = np.zeros_like(xns)
    
    xg, wg = generate_Gauss_quadrature(p)
    
    for i, xn in enumerate(xns):
        F_interval = 0
        for j in range(p):
            x = xn + 0.5 * (xg[j] + 1) * h
            y = func(x)
            F_interval += y * h * wg[j] * 0.5
        F[i] = F_interval
    
    return np.sum(F, keepdims = False)

# Function form for integration
def func_prob_a(x:float):
    return x ** 8

def int_func_prob_a(a:float, b:float):
    return (b ** 9 - a ** 9)/9

def func_prob_b(x:float):
    return abs(x-1/np.sqrt(2)) ** 3

def int_func_prob_b(a:float, b:float):
    return 0.25 * abs(b - 1 / np.sqrt(2)) ** 4 + 0.25 * abs(a - 1 / np.sqrt(2)) ** 4

def func_prob_c(x:float):
    return np.heaviside(x-1/np.sqrt(2),0)

def int_func_prob_c(a:float, b:float):
    return b - 1/np.sqrt(2) if b > 1/np.sqrt(2) else 0

def func_prob_d(x:float):
    return 1 / np.sqrt(x)

def int_func_prob_d(a:float, b:float):
    return 2 * np.sqrt(b) - 2*np.sqrt(a)

if __name__ == "__main__":

    # Parameters
    p = 5
    N_list = [2 * i for i in range(3,16)]

    # Exact values for function integral with prob (a) to (d)
    F_a = int_func_prob_a(-1,1)
    F_b = int_func_prob_b(-1,1)
    F_c = int_func_prob_c(-1,1)
    F_d = int_func_prob_d(0,1)

    h_a = []
    h_b = []
    h_c = []
    h_d = []

    err_a = []
    err_b = []
    err_c = []
    err_d = []

    for N in N_list:
        h_a.append(2/N)
        h_b.append(2/N)
        h_c.append(2/N)
        h_d.append(1/N)

        F_a_quad = compute_integral(-1,1,N,p,func_prob_a)
        F_b_quad = compute_integral(-1,1,N,p,func_prob_b)
        F_c_quad = compute_integral(-1,1,N,p,func_prob_c)
        F_d_quad = compute_integral(0,1,N,p,func_prob_d)

        err_a.append(compute_relative_err(F_a_quad,F_a))
        err_b.append(compute_relative_err(F_b_quad,F_b))
        err_c.append(compute_relative_err(F_c_quad,F_c))
        err_d.append(compute_relative_err(F_d_quad,F_d))
        
    fig, axes = plt.subplots(2,2,figsize = (10,6))

    axes[0,0].plot(h_a, err_a, 'r')
    axes[0,0].set_xlabel("h")
    axes[0,0].set_ylabel("Relative error")
    axes[0,0].set_title(r"$f(x)=x^8$ in [-1,1]")
    
    axes[0,1].plot(h_b, err_b, 'r')
    axes[0,1].set_xlabel("h")
    axes[0,1].set_ylabel("Relative error")
    axes[0,1].set_title(r"$f(x)=|x-1/\sqrt{2}|^3$ in [-1,1]")
    
    axes[1,0].plot(h_c, err_c, 'r')
    axes[1,0].set_xlabel("h")
    axes[1,0].set_ylabel("Relative error")
    axes[1,0].set_title(r"$f(x)=H(x-1/\sqrt{2})$ in [-1,1]")
    
    axes[1,1].plot(h_d, err_d, 'r')
    axes[1,1].set_xlabel("h")
    axes[1,1].set_ylabel("Relative error")
    axes[1,1].set_title(r"$f(x)=1/\sqrt{x}$ in [0,1]")
    
    fig.tight_layout()
    fig.savefig("./p2.png")