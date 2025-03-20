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

def compute_relative_err(pred:float,exact:float):
    err = abs(pred - exact) / (abs(exact) + 1e-8)
    return err

# Integration algorithm using Gaussian quadrature
def compute_integral(a:float, b:float, N:int, p:int, func:Callable):
    h = (b-a) / N
    xns = np.linspace(a,b,N, endpoint = True)
    F = np.zeros_like(xns)
    
    xg, wg = generate_Gauss_quadrature(p)
    
    for i, xn in enumerate(xns):
        F_interval = 0
        for j in range(p):
            x = xn + 0.5 * (xg[j] + 1) * h
            y = func(x)
            F_interval += y * h * wg[j] * 0.5
            
        F[i] = F_interval
    
    return np.sum(F[:-1], keepdims = False)

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
    N_list = [10 * i for i in range(1, 10)]

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

    # Power law observation
    h_a, h_b, h_c, h_d = np.array(h_a), np.array(h_b), np.array(h_c), np.array(h_d)
    err_a, err_b, err_c, err_d = np.array(err_a), np.array(err_b), np.array(err_c), np.array(err_d)

    p_err_a_h = np.polyfit(h_a ** 0.5, err_a, deg=1)
    p_err_b_h = np.polyfit(h_b ** 0.5, err_b, deg=1)
    p_err_c_h = np.polyfit(h_c ** 0.5, err_c, deg=1)
    p_err_d_h = np.polyfit(h_d ** 0.5, err_d, deg=1)

    line_err_a_h = np.poly1d(p_err_a_h)(h_a**0.5)
    line_err_b_h = np.poly1d(p_err_b_h)(h_b**0.5)
    line_err_c_h = np.poly1d(p_err_c_h)(h_c**0.5)
    line_err_d_h = np.poly1d(p_err_d_h)(h_d**0.5)

    p_err_a_1 = np.polyfit(h_a, err_a, deg=1)
    p_err_b_1 = np.polyfit(h_b, err_b, deg=1)
    p_err_c_1 = np.polyfit(h_c, err_c, deg=1)
    p_err_d_1 = np.polyfit(h_d, err_d, deg=1)

    line_err_a_1 = np.poly1d(p_err_a_1)(h_a)
    line_err_b_1 = np.poly1d(p_err_b_1)(h_b)
    line_err_c_1 = np.poly1d(p_err_c_1)(h_c)
    line_err_d_1 = np.poly1d(p_err_d_1)(h_d)

    p_err_a_2 = np.polyfit(h_a**2, err_a, deg=1)
    p_err_b_2 = np.polyfit(h_b**2, err_b, deg=1)
    p_err_c_2 = np.polyfit(h_c**2, err_c, deg=1)
    p_err_d_2 = np.polyfit(h_d**2, err_d, deg=1)

    line_err_a_2 = np.poly1d(p_err_a_2)(h_a**2)
    line_err_b_2 = np.poly1d(p_err_b_2)(h_b**2)
    line_err_c_2 = np.poly1d(p_err_c_2)(h_c**2)
    line_err_d_2 = np.poly1d(p_err_d_2)(h_d**2)

    # Estimation of power law by linear fitting with log-log scale
    p_err_a = np.polyfit(np.log(h_a), np.log(err_a), deg=1)
    p_err_b = np.polyfit(np.log(h_b), np.log(err_b), deg=1)
    p_err_c = np.polyfit(np.log(h_c), np.log(err_c), deg=1)
    p_err_d = np.polyfit(np.log(h_d), np.log(err_d), deg=1)

    line_err_a = np.exp(np.poly1d(p_err_a)(np.log(h_a)))
    line_err_b = np.exp(np.poly1d(p_err_b)(np.log(h_b)))
    line_err_c = np.exp(np.poly1d(p_err_c)(np.log(h_c)))
    line_err_d = np.exp(np.poly1d(p_err_d)(np.log(h_d)))

    print("p_err_a:{:.3f}".format(p_err_a[0]))
    print("p_err_b:{:.3f}".format(p_err_b[0]))
    print("p_err_c:{:.3f}".format(p_err_c[0]))
    print("p_err_d:{:.3f}".format(p_err_d[0]))

    fig, axes = plt.subplots(2,2,figsize = (10,8))

    axes[0,0].plot(h_a, err_a, 'r', label = "Error")
    axes[0,0].plot(h_a, line_err_a_h, 'g--', label = "$h^{0.5}$")
    axes[0,0].plot(h_a, line_err_a_1, 'k--', label = "$h^{1}$")
    axes[0,0].plot(h_a, line_err_a_2, 'b--', label = "$h^{2}$")
    axes[0,0].plot(h_a, line_err_a, "c--", label=r"$h^{{{:.3f}}}$".format(p_err_a[0]))
    axes[0,0].set_xlabel("h")
    axes[0,0].set_ylabel("Relative error")
    axes[0,0].set_title(r"$f(x)=x^8$ in [-1,1]")
    axes[0,0].legend(loc = "lower right")

    axes[0,1].plot(h_b, err_b, 'r', label = "Error")
    axes[0,1].plot(h_b, line_err_b_h, 'g--', label = "$h^{0.5}$")
    axes[0,1].plot(h_b, line_err_b_1, 'k--', label = "$h^{1}$")
    axes[0,1].plot(h_b, line_err_b_2, 'b--', label = "$h^{2}$")
    axes[0,1].plot(h_b, line_err_b, "c--", label=r"$h^{{{:.3f}}}$".format(p_err_b[0]))
    axes[0,1].set_xlabel("h")
    axes[0,1].set_ylabel("Relative error")
    axes[0,1].set_title(r"$f(x)=|x-1/\sqrt{2}|^3$ in [-1,1]")
    axes[0,1].legend(loc = "lower right")

    axes[1,0].plot(h_c, err_c, 'r', label = "Error")
    axes[1,0].plot(h_c, line_err_c_h, 'g--', label = "$h^{0.5}$")
    axes[1,0].plot(h_c, line_err_c_1, 'k--', label = "$h^{1}$")
    axes[1,0].plot(h_c, line_err_c_2, 'b--', label = "$h^{2}$")
    axes[1,0].plot(h_c, line_err_c, "c--", label=r"$h^{{{:.3f}}}$".format(p_err_c[0]))
    axes[1,0].set_xlabel("h")
    axes[1,0].set_ylabel("Relative error")
    axes[1,0].set_title(r"$f(x)=H(x-1/\sqrt{2})$ in [-1,1]")
    axes[1,0].legend(loc = "lower right")

    axes[1,1].plot(h_d, err_d, 'r', label = "Error")
    axes[1,1].plot(h_d, line_err_d_h, 'g--', label = "$h^{0.5}$")
    axes[1,1].plot(h_d, line_err_d_1, 'k--', label = "$h^{1}$")
    axes[1,1].plot(h_d, line_err_d_2, 'b--', label = "$h^{2}$")
    axes[1,1].plot(h_d, line_err_d, "c--", label=r"$h^{{{:.3f}}}$".format(p_err_d[0]))
    axes[1,1].set_xlabel("h")
    axes[1,1].set_ylabel("Relative error")
    axes[1,1].set_title(r"$f(x)=1/\sqrt{x}$ in [0,1]")
    axes[1,1].legend(loc = "lower right")

    fig.tight_layout()
    fig.savefig("./p2.png")

    # Problem 2-bonus
    # Split the interval
    # Since f(x < 1/np.sqrt(2)) = 0, we only need to consider the integration domain where x > np.sqrt(2)
    # F_c_quad_p1 : integration of x in [-1, 1/np.sqrt(2)] => 0 for all cases
    F_c = int_func_prob_c(-1, 1)

    h_c_split = []
    err_c_split = []

    N_list = [5 * i for i in range(1, 10)]

    for N in N_list:
        F_c_quad_p1 = 0
        F_c_quad_p2 = compute_integral(1/np.sqrt(2), 1, N, p, func_prob_c)

        F_c_quad = F_c_quad_p1 + F_c_quad_p2 

        h_c_split.append((1-1/np.sqrt(2))/N)
        err_c_split.append(compute_relative_err(F_c_quad, F_c))

    h_c_split = np.array(h_c_split)
    err_c_split = np.array(err_c_split)

    p_err_c_split = np.polyfit(np.log(h_c_split), np.log(err_c_split), deg=1)
    line_err_c_split = np.exp(np.poly1d(p_err_c_split)(np.log(h_c_split)))

    fig, ax = plt.subplots(1,1,figsize = (6,4))

    ax.plot(h_c_split, err_c_split, 'r', label = "Error w/ split")
    ax.plot(h_c, err_c, "b", label="Error w/o split")
    ax.plot(h_c_split, line_err_c_split, "k--", label=r"$h^{{{:.3f}}}$".format(p_err_c_split[0]))
    ax.plot(h_c, line_err_c, "c--", label=r"$h^{{{:.3f}}}$".format(p_err_c[0]))
    ax.set_xlabel("h")
    ax.set_ylabel("Relative error")
    ax.set_xlim([h_c.min(), h_c.max()])
    ax.set_title(r"$f(x)=H(x-1/\sqrt{2})$ in [-1,1]")
    ax.legend(loc = "upper right")
    fig.tight_layout()
    fig.savefig("./p2_bonus.png")
