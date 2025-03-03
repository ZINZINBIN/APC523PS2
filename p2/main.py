"""
    Problem 2. Integration
        Write a numerical integration function that takes the following arguments: a function f(x), 
        a number of discrete intervals N, and the integration bounds aand b. The function must return 
        the value of integral f(x) from a to b. The integrator should use 5-point Gaussian Quadrature 
        method in each of the N intervals.
        
        Plot the relative error of integration as a function of h= (b−a)/N for
        (a) a=−1,b= 1,f(x) = x8
        (b) a=−1,b= 1,f(x) = x−1/√2 3
        (c) a=−1,b= 1,f(x) = H x−1/√2 ,
        (d) a= 0,b= 1,f(x) = 1/√x.
        
        Determine the power law dependency of each error curve as a function of h(e.g. h^0.5, h^1, or h^2).
        Then, plot the corresponding power-law trend as a dashed line for reference. Write the explanations
        of the power law dependencies that you find for each function in the write-up.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pass