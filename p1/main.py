"""
    Problem 1. Polynomial Interpolation
        Implement a method using Lagrange polynomial interpolation of degree pusing for the p+ 1 node
        points the roots of the degree p+ 1 Chebyshev polynomial. Write a function that takes as input 
        f(x) on the interval [-1,1] and interpolates it to g(x). Compute the L2 error and the maximum
        error of your interpolation using a fine grid with N = 1000 points uniformly distributed in the
        domain.
        
        (1) Plot the actual function f(x) and its approximation g(x) as function of xon the fine grid. 
        Do this for each function above using 3 different Lagrange polynomial degrees p= {10,20,40}.
        Mark the p+ 1 Chebyshev nodes on the plot of g(x).
        
        (2) Plot the L2 and maximum errors as a function of p∈[1,256] using a log-log scale. 
        Label the observed power law dependencies using dashed lines. Interpret the results.
        
        (Bonus) Try and go up to p = 1024. You will need to take advantage of the property that 
        the Lagrange polynomials based on Chebyshev nodes can be written using Chebyshev polynomials, 
        which are easier and faster to compute.
"""

import numpy as np
import matplotlib.pyplot as plt

# evaluate error (L2 norm)
def compute_l2_err(f:np.ndarray,g:np.ndarray):
    # err = np.linalg.norm(f-g, keepdims=False)
    err = np.sqrt(np.sum((f-g)**2) / len(f))
    return err

# evaluate error (inf norm)
def compute_max_err(f:np.ndarray,g:np.ndarray):
    err = max(abs(f-g))
    return err

# chebyshev nodes generator
def generate_cheb_nodes(xmin:float = -1.0, xmax:float = 1.0, p:int = 10):
    x = []
    
    for i in range(0, p + 1):
        node = (xmax + xmin)/2 + (xmax-xmin)/2 * np.cos((2 * i + 1) / (2 * p + 2) * np.pi)
        x.append(node)
        
    return np.array(x)

# exact function form given in problem 1
def func_prob_a(x:float):
    return 1 / (1+25.0 * x ** 2)

def func_prob_b(x:float):
    return abs(x)

def func_prob_c(x:float):
    return np.heaviside(x-0.25,0)

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

def generate_pth_cheb_polynomial(x:np.array, p:int):
    
    T = np.ones((p + 1, len(x)))
    
    if p==0:
        return T
    
    T[1] = x

    for n in range(1,p):
        T[n+1] = 2 * x * T[n] - T[n-1]

    return T

def generate_cheb_polynomial(x:np.array, y:np.array, p:int):

    T = generate_pth_cheb_polynomial(x,p)
    a = np.zeros(p+1)
    a[0] = np.mean(y)
    
    for i in range(1,p+1):
        a[i] = 2 / len(x) * np.sum(y * T[i])
    
    def interp(xn:np.array):
        Tn = generate_pth_cheb_polynomial(xn, p)
        r = np.sum(a.reshape(-1,1) * Tn, axis = 0, keepdims = False) 
        return r
    
    return interp

if __name__ == "__main__":

    # Parameters
    N = 1000
    xmin = -1.0
    xmax = 1.0
    p_list = [10,20,40]

    x_grid = np.linspace(xmin, xmax, N, endpoint = True)

    # Save results
    x_node_list = []
    y_node_a_list = []
    y_node_b_list = []
    y_node_c_list = []

    y_grid_a_list = []
    y_grid_b_list = []
    y_grid_c_list = []

    y_interp_a_list = []
    y_interp_b_list = []
    y_interp_c_list = []

    # Problem (1)
    for p in p_list:

        x_node = generate_cheb_nodes(xmin, xmax, p)

        y_node_a = func_prob_a(x_node)
        y_node_b = func_prob_b(x_node)
        y_node_c = func_prob_c(x_node)

        pol_a = generate_langrange_polynomial(x_node, y_node_a)
        pol_b = generate_langrange_polynomial(x_node, y_node_b)
        pol_c = generate_langrange_polynomial(x_node, y_node_c)

        y_grid_a = func_prob_a(x_grid)
        y_grid_b = func_prob_b(x_grid)
        y_grid_c = func_prob_c(x_grid)

        y_interp_a = pol_a(x_grid)
        y_interp_b = pol_b(x_grid)
        y_interp_c = pol_c(x_grid)    

        # save
        x_node_list.append(x_node)
        y_node_a_list.append(y_node_a)
        y_node_b_list.append(y_node_b)
        y_node_c_list.append(y_node_c)

        y_grid_a_list.append(y_grid_a)
        y_grid_b_list.append(y_grid_b)
        y_grid_c_list.append(y_grid_c)

        y_interp_a_list.append(y_interp_a)
        y_interp_b_list.append(y_interp_b)
        y_interp_c_list.append(y_interp_c)

        # compute error
        l2_err_a = compute_l2_err(y_grid_a, y_interp_a)
        l2_err_b = compute_l2_err(y_grid_b, y_interp_b)
        l2_err_c = compute_l2_err(y_grid_c, y_interp_c)

        inf_err_a = compute_max_err(y_grid_a, y_interp_a)
        inf_err_b = compute_max_err(y_grid_b, y_interp_b)
        inf_err_c = compute_max_err(y_grid_c, y_interp_c)

        print("p = {}, l2 error (a):{:.3f}".format(p, l2_err_a))
        print("p = {}, l2 error (b):{:.3f}".format(p, l2_err_b))
        print("p = {}, l2 error (c):{:.3f}".format(p, l2_err_c))
        
        print("p = {}, Max error (a):{:.3f}".format(p, inf_err_a))
        print("p = {}, Max error (b):{:.3f}".format(p, inf_err_b))
        print("p = {}, Max error (c):{:.3f}".format(p, inf_err_c))

    fig, axes = plt.subplots(3,3,figsize = (12,9))

    axes[0,0].scatter(x_node_list[0], y_node_a_list[0], c = 'k', marker = 'o')
    axes[0,0].plot(x_grid, y_grid_a_list[0], 'r', label = "exact $f(x)$")
    axes[0,0].plot(x_grid, y_interp_a_list[0], 'b-', label = "interpol")
    axes[0,0].set_xlabel("$x$")
    axes[0,0].set_ylabel("$f(x)$")
    axes[0,0].set_title("$f(x)= 1/(1+25x^2)$ with p = {}".format(p_list[0]))
    axes[0,0].legend(loc = "upper right")

    axes[0,1].scatter(x_node_list[0], y_node_b_list[0], c = 'k', marker = 'o')
    axes[0,1].plot(x_grid, y_grid_b_list[0], 'r', label = "exact $f(x)$")
    axes[0,1].plot(x_grid, y_interp_b_list[0], 'b-', label = "interpol")
    axes[0,1].set_xlabel("$x$")
    axes[0,1].set_ylabel("$f(x)$")
    axes[0,1].set_title("$f(x)= |x|$ with p = {}".format(p_list[0]))
    axes[0,1].legend(loc = "upper center")

    axes[0,2].scatter(x_node_list[0], y_node_c_list[0], c = 'k', marker = 'o')
    axes[0,2].plot(x_grid, y_grid_c_list[0], 'r', label = "exact $f(x)$")
    axes[0,2].plot(x_grid, y_interp_c_list[0], 'b-', label = "interpol")
    axes[0,2].set_xlabel("$x$")
    axes[0,2].set_ylabel("$f(x)$")
    axes[0,2].set_title("$f(x)= H(x-0.25)$ with p = {}".format(p_list[0]))
    axes[0,2].legend(loc = "upper left")

    axes[1,0].scatter(x_node_list[1], y_node_a_list[1], c = 'k', marker = 'o')
    axes[1,0].plot(x_grid, y_grid_a_list[1], 'r', label = "exact $f(x)$")
    axes[1,0].plot(x_grid, y_interp_a_list[1], 'b-', label = "interpol")
    axes[1,0].set_xlabel("$x$")
    axes[1,0].set_ylabel("$f(x)$")
    axes[1,0].set_title("$f(x)= 1/(1+25x^2)$ with p = {}".format(p_list[1]))
    axes[1,0].legend(loc = "upper right")

    axes[1,1].scatter(x_node_list[1], y_node_b_list[1], c = 'k', marker = 'o')
    axes[1,1].plot(x_grid, y_grid_b_list[1], 'r', label = "exact $f(x)$")
    axes[1,1].plot(x_grid, y_interp_b_list[1], 'b-', label = "interpol")
    axes[1,1].set_xlabel("$x$")
    axes[1,1].set_ylabel("$f(x)$")
    axes[1,1].set_title("$f(x)= |x|$ with p = {}".format(p_list[1]))
    axes[1,1].legend(loc = "upper center")

    axes[1,2].scatter(x_node_list[1], y_node_c_list[1], c = 'k', marker = 'o')
    axes[1,2].plot(x_grid, y_grid_c_list[1], 'r', label = "exact $f(x)$")
    axes[1,2].plot(x_grid, y_interp_c_list[1], 'b-', label = "interpol")
    axes[1,2].set_xlabel("$x$")
    axes[1,2].set_ylabel("$f(x)$")
    axes[1,2].set_title("$f(x)= H(x-0.25)$ with p = {}".format(p_list[1]))
    axes[1,2].legend(loc = "upper left")

    axes[2,0].scatter(x_node_list[2], y_node_a_list[2], c = 'k', marker = 'o')
    axes[2,0].plot(x_grid, y_grid_a_list[2], 'r', label = "exact $f(x)$")
    axes[2,0].plot(x_grid, y_interp_a_list[2], 'b-', label = "interpol")
    axes[2,0].set_xlabel("$x$")
    axes[2,0].set_ylabel("$f(x)$")
    axes[2,0].set_title("$f(x)= 1/(1+25x^2)$ with p = {}".format(p_list[2]))
    axes[2,0].legend(loc = "upper right")

    axes[2,1].scatter(x_node_list[2], y_node_b_list[2], c = 'k', marker = 'o')
    axes[2,1].plot(x_grid, y_grid_b_list[2], 'r', label = "exact $f(x)$")
    axes[2,1].plot(x_grid, y_interp_b_list[2], 'b-', label = "interpol")
    axes[2,1].set_xlabel("$x$")
    axes[2,1].set_ylabel("$f(x)$")
    axes[2,1].set_title("$f(x)= |x|$ with p = {}".format(p_list[2]))
    axes[2,1].legend(loc = "upper center")

    axes[2,2].scatter(x_node_list[2], y_node_c_list[2], c = 'k', marker = 'o')
    axes[2,2].plot(x_grid, y_grid_c_list[2], 'r', label = "exact $f(x)$")
    axes[2,2].plot(x_grid, y_interp_c_list[2], 'b-', label = "interpol")
    axes[2,2].set_xlabel("$x$")
    axes[2,2].set_ylabel("$f(x)$")
    axes[2,2].set_title("$f(x)= H(x-0.25)$ with p = {}".format(p_list[2]))
    axes[2,2].legend(loc = "upper left")

    fig.tight_layout()
    fig.savefig("./p1_1.png")

    # Problem (2)
    p_list = np.arange(4, 256, 16)

    l2_err_a_list = []
    l2_err_b_list = []
    l2_err_c_list = []

    inf_err_a_list = []
    inf_err_b_list = []
    inf_err_c_list = []

    for p in p_list:
        x_node = generate_cheb_nodes(xmin, xmax, p)

        y_node_a = func_prob_a(x_node)
        y_node_b = func_prob_b(x_node)
        y_node_c = func_prob_c(x_node)

        pol_a = generate_langrange_polynomial(x_node, y_node_a)
        pol_b = generate_langrange_polynomial(x_node, y_node_b)
        pol_c = generate_langrange_polynomial(x_node, y_node_c)

        y_grid_a = func_prob_a(x_grid)
        y_grid_b = func_prob_b(x_grid)
        y_grid_c = func_prob_c(x_grid)

        y_interp_a = pol_a(x_grid)
        y_interp_b = pol_b(x_grid)
        y_interp_c = pol_c(x_grid)

        l2_err_a = compute_l2_err(y_grid_a, y_interp_a)  
        l2_err_b = compute_l2_err(y_grid_b, y_interp_b)
        l2_err_c = compute_l2_err(y_grid_c, y_interp_c)

        inf_err_a = compute_max_err(y_grid_a, y_interp_a)
        inf_err_b = compute_max_err(y_grid_b, y_interp_b)
        inf_err_c = compute_max_err(y_grid_c, y_interp_c)

        # save
        l2_err_a_list.append(l2_err_a)
        l2_err_b_list.append(l2_err_b)
        l2_err_c_list.append(l2_err_c)

        inf_err_a_list.append(inf_err_a)
        inf_err_b_list.append(inf_err_b)
        inf_err_c_list.append(inf_err_c)

    # Power law observation
    p_l2_err_a = np.polyfit(np.log(p_list), np.log(l2_err_a_list), deg = 1)
    p_l2_err_b = np.polyfit(np.log(p_list), np.log(l2_err_b_list), deg = 1)
    p_l2_err_c = np.polyfit(np.log(p_list), np.log(l2_err_c_list), deg = 1)

    line_l2_err_a = np.poly1d(p_l2_err_a)(np.log(p_list))
    line_l2_err_b = np.poly1d(p_l2_err_b)(np.log(p_list))
    line_l2_err_c = np.poly1d(p_l2_err_c)(np.log(p_list))

    p_inf_err_a = np.polyfit(np.log(p_list), np.log(inf_err_a_list), deg = 1)
    p_inf_err_b = np.polyfit(np.log(p_list), np.log(inf_err_b_list), deg = 1)
    p_inf_err_c = np.polyfit(np.log(p_list), np.log(inf_err_c_list), deg = 1)

    line_inf_err_a = np.poly1d(p_inf_err_a)(np.log(p_list))
    line_inf_err_b = np.poly1d(p_inf_err_b)(np.log(p_list))
    line_inf_err_c = np.poly1d(p_inf_err_c)(np.log(p_list))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0,0].plot(p_list, l2_err_a_list, 'r', label = "Error")
    axes[0,0].plot(p_list, np.exp(line_l2_err_a), 'k--', label = r"Power law $O(p^{{{:.3f}}})$".format(p_l2_err_a[0]))
    axes[0,0].set_xlabel("$p$")
    axes[0,0].set_ylabel("$L2$ error")
    axes[0,0].set_title("$f(x)= 1/(1+25x^2)$")
    axes[0,0].set_xscale("log")
    axes[0,0].set_yscale("log")
    axes[0,0].legend(loc = "upper right")

    axes[0,1].plot(p_list, l2_err_b_list, 'r', label = "Error")
    axes[0,1].plot(p_list, np.exp(line_l2_err_b), 'k--', label = r"Power law $O(p^{{{:.3f}}})$".format(p_l2_err_b[0]))
    axes[0,1].set_xlabel("$p$")
    axes[0,1].set_ylabel("$L2$ error")
    axes[0,1].set_title("$f(x)= |x|$")
    axes[0,1].set_xscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].legend(loc = "upper right")

    axes[0,2].plot(p_list, l2_err_c_list, 'r', label = "Error")
    axes[0,2].plot(p_list, np.exp(line_l2_err_c), 'k--', label = r"Power law $O(p^{{{:.3f}}})$".format(p_l2_err_c[0]))
    axes[0,2].set_xlabel("$x$")
    axes[0,2].set_ylabel("$L2$ error")
    axes[0,2].set_title("$f(x)= H(x-0.25)$")
    axes[0,2].set_xscale("log")
    axes[0,2].set_yscale("log")
    axes[0,2].legend(loc = "upper right")

    axes[1,0].plot(p_list, inf_err_a_list, 'r', label = "Error")
    axes[1,0].plot(p_list, np.exp(line_inf_err_a), 'k--', label = r"Power law $O(p^{{{:.3f}}})$".format(p_inf_err_a[0]))
    axes[1,0].set_xlabel("$p$")
    axes[1,0].set_ylabel("Max error")
    axes[1,0].set_title("$f(x)= 1/(1+25x^2)$")
    axes[1,0].set_xscale("log")
    axes[1,0].set_yscale("log")
    axes[1,0].legend(loc = "upper right")

    axes[1,1].plot(p_list, inf_err_b_list, 'r', label = "Error")
    axes[1,1].plot(p_list, np.exp(line_inf_err_b), 'k--', label = r"Power law $O(p^{{{:.3f}}})$".format(p_inf_err_b[0]))
    axes[1,1].set_xlabel("$p$")
    axes[1,1].set_ylabel("Max error")
    axes[1,1].set_title("$f(x)= |x|$")
    axes[1,1].set_xscale("log")
    axes[1,1].set_yscale("log")
    axes[1,1].legend(loc = "upper right")

    axes[1,2].plot(p_list, inf_err_c_list, 'r', label = "Error")
    axes[1,2].plot(p_list, np.exp(line_inf_err_c), 'k--', label = r"Power law $O(p^{{{:.3f}}})$".format(p_inf_err_c[0]))
    axes[1,2].set_xlabel("$p$")
    axes[1,2].set_ylabel("Max error")
    axes[1,2].set_title("$f(x)= H(x-0.25)$")
    axes[1,2].set_xscale("log")
    axes[1,2].set_yscale("log")
    axes[1,2].legend(loc = "upper right")

    fig.tight_layout()
    fig.savefig("./p1_2.png")

    # Bonus problem. Chebyshev polynomial with p = 1024
    p = 1024
    x_node = generate_cheb_nodes(xmin, xmax, p)

    y_node_a = func_prob_a(x_node)
    y_node_b = func_prob_b(x_node)
    y_node_c = func_prob_c(x_node)

    # generate chebyshev polynomial
    pol_a = generate_cheb_polynomial(x_node, y_node_a, p)
    pol_b = generate_cheb_polynomial(x_node, y_node_b, p)
    pol_c = generate_cheb_polynomial(x_node, y_node_c, p)

    y_grid_a = func_prob_a(x_grid)
    y_grid_b = func_prob_b(x_grid)
    y_grid_c = func_prob_c(x_grid)

    y_interp_a = pol_a(x_grid)
    y_interp_b = pol_b(x_grid)
    y_interp_c = pol_c(x_grid)    

    # plot the result
    fig, axes = plt.subplots(1,3,figsize = (12,4))
    axes = axes.ravel()

    axes[0].scatter(x_node, y_node_a, c = 'k', s = 0.1, marker = 'o')
    axes[0].plot(x_grid, y_grid_a, 'r', label = "exact $f(x)$")
    axes[0].plot(x_grid, y_interp_a, 'b-', label = "interpol")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$f(x)$")
    axes[0].set_title("$f(x)= 1/(1+25x^2)$ with p = {}".format(p))
    axes[0].legend(loc = "upper right")

    axes[1].scatter(x_node, y_node_b, c="k", s=0.1, marker="o")
    axes[1].plot(x_grid, y_grid_b, 'r', label = "exact $f(x)$")
    axes[1].plot(x_grid, y_interp_b, 'b-', label = "interpol")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$f(x)$")
    axes[1].set_title("$f(x)= |x|$ with p = {}".format(p))
    axes[1].legend(loc = "upper center")

    axes[2].scatter(x_node, y_node_c, c="k", s=0.1, marker="o")
    axes[2].plot(x_grid, y_grid_c, 'r', label = "exact $f(x)$")
    axes[2].plot(x_grid, y_interp_c, 'b-', label = "interpol")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$f(x)$")
    axes[2].set_title("$f(x)= H(x-0.25)$ with p = {}".format(p))
    axes[2].legend(loc = "upper left")

    fig.tight_layout()
    fig.savefig("./p1_bonus_1.png")

    # Again, the error analysis for p in [4,1024]
    p_list = [2 ** i for i in range(2,11)]

    l2_err_a_list = []
    l2_err_b_list = []
    l2_err_c_list = []

    inf_err_a_list = []
    inf_err_b_list = []
    inf_err_c_list = []

    for p in p_list:
        x_node = generate_cheb_nodes(xmin, xmax, p)

        y_node_a = func_prob_a(x_node)
        y_node_b = func_prob_b(x_node)
        y_node_c = func_prob_c(x_node)

        pol_a = generate_cheb_polynomial(x_node, y_node_a, p)
        pol_b = generate_cheb_polynomial(x_node, y_node_b, p)
        pol_c = generate_cheb_polynomial(x_node, y_node_c, p)

        y_grid_a = func_prob_a(x_grid)
        y_grid_b = func_prob_b(x_grid)
        y_grid_c = func_prob_c(x_grid)

        y_interp_a = pol_a(x_grid)
        y_interp_b = pol_b(x_grid)
        y_interp_c = pol_c(x_grid)

        l2_err_a = compute_l2_err(y_grid_a, y_interp_a)
        l2_err_b = compute_l2_err(y_grid_b, y_interp_b)
        l2_err_c = compute_l2_err(y_grid_c, y_interp_c)

        inf_err_a = compute_max_err(y_grid_a, y_interp_a)
        inf_err_b = compute_max_err(y_grid_b, y_interp_b)
        inf_err_c = compute_max_err(y_grid_c, y_interp_c)

        # save
        l2_err_a_list.append(l2_err_a)
        l2_err_b_list.append(l2_err_b)
        l2_err_c_list.append(l2_err_c)

        inf_err_a_list.append(inf_err_a)
        inf_err_b_list.append(inf_err_b)
        inf_err_c_list.append(inf_err_c)

    # Power law observation
    p_l2_err_a = np.polyfit(np.log(p_list), np.log(l2_err_a_list), deg=1)
    p_l2_err_b = np.polyfit(np.log(p_list), np.log(l2_err_b_list), deg=1)
    p_l2_err_c = np.polyfit(np.log(p_list), np.log(l2_err_c_list), deg=1)

    line_l2_err_a = np.poly1d(p_l2_err_a)(np.log(p_list))
    line_l2_err_b = np.poly1d(p_l2_err_b)(np.log(p_list))
    line_l2_err_c = np.poly1d(p_l2_err_c)(np.log(p_list))

    p_inf_err_a = np.polyfit(np.log(p_list), np.log(inf_err_a_list), deg=1)
    p_inf_err_b = np.polyfit(np.log(p_list), np.log(inf_err_b_list), deg=1)
    p_inf_err_c = np.polyfit(np.log(p_list), np.log(inf_err_c_list), deg=1)

    line_inf_err_a = np.poly1d(p_inf_err_a)(np.log(p_list))
    line_inf_err_b = np.poly1d(p_inf_err_b)(np.log(p_list))
    line_inf_err_c = np.poly1d(p_inf_err_c)(np.log(p_list))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].plot(p_list, l2_err_a_list, "r", label="Error")
    axes[0, 0].plot(p_list, np.exp(line_l2_err_a), "k--", label = r"Power law $O(p^{{{:.3f}}})$".format(p_l2_err_a[0]))
    axes[0, 0].set_xlabel("$p$")
    axes[0, 0].set_ylabel("$L2$ error")
    axes[0, 0].set_title("$f(x)= 1/(1+25x^2)$")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(p_list, l2_err_b_list, "r", label="Error")
    axes[0, 1].plot(p_list, np.exp(line_l2_err_b), "k--", label = r"Power law $O(p^{{{:.3f}}})$".format(p_l2_err_b[0]))
    axes[0, 1].set_xlabel("$p$")
    axes[0, 1].set_ylabel("$L2$ error")
    axes[0, 1].set_title("$f(x)= |x|$")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(loc="upper right")

    axes[0, 2].plot(p_list, l2_err_c_list, "r", label="Error")
    axes[0, 2].plot(p_list, np.exp(line_l2_err_c), "k--", label = r"Power law $O(p^{{{:.3f}}})$".format(p_l2_err_c[0]))
    axes[0, 2].set_xlabel("$x$")
    axes[0, 2].set_ylabel("$L2$ error")
    axes[0, 2].set_title("$f(x)= H(x-0.25)$")
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].legend(loc="upper right")

    axes[1, 0].plot(p_list, inf_err_a_list, "r", label="Error")
    axes[1, 0].plot(p_list, np.exp(line_inf_err_a), "k--", label = r"Power law $O(p^{{{:.3f}}})$".format(p_inf_err_a[0]))
    axes[1, 0].set_xlabel("$p$")
    axes[1, 0].set_ylabel("Max error")
    axes[1, 0].set_title("$f(x)= 1/(1+25x^2)$")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(p_list, inf_err_b_list, "r", label="Error")
    axes[1, 1].plot(p_list, np.exp(line_inf_err_b), "k--", label = r"Power law $O(p^{{{:.3f}}})$".format(p_inf_err_b[0]))
    axes[1, 1].set_xlabel("$p$")
    axes[1, 1].set_ylabel("Max error")
    axes[1, 1].set_title("$f(x)= |x|$")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(loc="upper right")

    axes[1, 2].plot(p_list, inf_err_c_list, "r", label="Error")
    axes[1, 2].plot(p_list, np.exp(line_inf_err_c), "k--", label = r"Power law $O(p^{{{:.3f}}})$".format(p_inf_err_c[0]))
    axes[1, 2].set_xlabel("$p$")
    axes[1, 2].set_ylabel("Max error")
    axes[1, 2].set_title("$f(x)= H(x-0.25)$")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig("./p1_bonus_2.png")
