"""
    Problem 3. Iterative Method
        Consider now the specific case α = 2. The matrix T2 now corresponds to the second-order finite difference 
        approximation of Laplace equation. We want to solve the linear system
        
            T2x = λmvm (7)
            
        where vm is a sine mode. Since a sine mode is a linear combination of two Fourier modes with eigenvalue λm, 
        it is also an eigenvector and therefore is a trivial solution of equation (7). Implement the Jacobi iterative 
        scheme for 1000 iterations to solve for x (using N = 100) starting from x(0) = 0. Define the error of kth iteration 
        as 
            e(k) = max|x(∞)−x(k)| (8)
            
        where xn is the nth component of vector x and x(∞) = vm is the solution for k→∞.
        
        (1) Plot the error as a function of the iteration count for eigenvectors v1 and v2. What are the observed convergence rates?
        (2) How do the observed convergence rates compared to those you derived earlier?
        (3) Based on your findings, what do you expect the convergence rate will be for an arbitrary vector y= 0≤m≤N−1 ymvm?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

def compute_iteration_error(x_pred:np.ndarray, x_target:np.ndarray):
    err = np.max(np.abs(x_pred - x_target)).item()
    return err

def compute_fro_norm(A:np.ndarray, x:np.ndarray, lamda:float):
    err = np.linalg.norm(A@x - lamda * x, ord = 'fro', keepdims = False)
    return err

def generate_eigen_vec(m:int, N:int):
    v = np.array([np.exp(-1j * 2 * np.pi * i * m / N) for i in range(0,N)]).reshape(-1,1)
    return v

def generate_eigen_value(alpha:float, m:int, N:int):
    return alpha - 2 * np.cos(2 * np.pi * m / N)

def generate_Ta(alpha:float, N:int):
    Ta = lil_matrix((N, N))
    coeffs = [-1.0, alpha, -1.0]
    for offset, coeff in zip([-1, 0, 1], coeffs):
        Ta.setdiag(coeff, offset)

    M = np.eye(N) * alpha
    M_inv = np.eye(N) * 1 / alpha
    G = M_inv @ (M - Ta)
    return Ta, M_inv, G

if __name__ == "__main__":

    # Parameters
    n_iters = 1000
    N = 100
    alpha = 2.0
    x0 = np.zeros((N,1))
    x = np.copy(x0)

    Ta, M_inv, G = generate_Ta(alpha, N)

    # Jacobi iteration with m = 1
    v1 = np.imag(generate_eigen_vec(m = 1, N = N))
    l1 = generate_eigen_value(alpha, m = 1, N = N)
    b = l1 * v1
    Mv = M_inv @ b

    v1_err_list = []
    v1_fro_err_list = []

    for n_iter in range(n_iters):
        err = compute_iteration_error(x, v1)
        v1_err_list.append(err)
        v1_fro_err_list.append(compute_fro_norm(Ta, x, l1))
        x = G@x + Mv

    v1_Jacobi = np.copy(x)

    # Jacobi iteration with m = 2
    x = np.copy(x0)
    v2 = np.imag(generate_eigen_vec(m=2, N=N))
    l2 = generate_eigen_value(alpha, m=2, N=N)
    b = l2 * v2
    Mv = M_inv @ b

    v2_err_list = []
    v2_fro_err_list = []

    for n_iter in range(n_iters):
        err = compute_iteration_error(x, v2)
        v2_err_list.append(err)
        v2_fro_err_list.append(compute_fro_norm(Ta, x, l2))
        x = G @ x + Mv

    v2_Jacobi = np.copy(x)

    x_iters = np.array([i + 1 for i in range(n_iters)])
    v1_err = np.array(v1_err_list)
    v2_err = np.array(v2_err_list)

    p_v1 = np.polyfit(x_iters, np.log(v1_err), deg = 1)
    p_v2 = np.polyfit(x_iters, np.log(v2_err), deg = 1)

    rho_v1_observed = np.exp(p_v1[0])
    rho_v2_observed = np.exp(p_v2[0])

    # log(eps) / log(rho) = n
    print("Spectral radius observed in v1:",rho_v1_observed)
    print("Spectral radius observed in v2:",rho_v2_observed)

    # Plot Max error for problem (1)
    fig, axes = plt.subplots(1,2,figsize = (8,4))
    axes = axes.ravel()

    axes[0].plot(x_iters, v1_err, 'r')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Max error")
    axes[0].set_title(r"Jacobi iteration with eigenvector m = 1")

    axes[1].plot(x_iters, v2_err, "r")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Max error")
    axes[1].set_title(r"Jacobi iteration with eigenvector m = 2")

    fig.tight_layout()
    fig.savefig("./p3.png")

    # Plot Fro norm for checking Av = lv
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.ravel()

    axes[0].plot(x_iters, v1_fro_err_list, "r")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("$|T_2 v_m - \\lambda_m v_m|_F$")
    axes[0].set_title(r"Jacobi iteration with eigenvector m = 1")

    axes[1].plot(x_iters, v2_fro_err_list, "r")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("$|T_2 v_m - \\lambda_m v_m|_F$")
    axes[1].set_title(r"Jacobi iteration with eigenvector m = 2")

    fig.tight_layout()
    fig.savefig("./p3_fro.png")

    # Arbitrary vector y to verify its convergence
    y = 0
    a = np.zeros(N)
    y_err_list = []
    y_fro_err_list = []

    for i in range(N):
        a[i] = np.random.rand()
        y += np.imag(generate_eigen_vec(m=i, N=N)) * a[i]

    # Use Jacobi iteration to solve Tx = y
    b = y
    Mv = M_inv @ b

    x = np.copy(x0)
    for n_iter in range(n_iters):
        err = compute_iteration_error(x, y)
        fro_err = np.linalg.norm(Ta@x-y, ord = 'fro', keepdims = False)
        y_err_list.append(err)
        y_fro_err_list.append(fro_err)
        x = G @ x + Mv

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.ravel()

    axes[0].plot(x_iters, y_err_list, "r")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Max error")
    axes[0].set_title(r"Jacobi iteration with $y$")

    axes[1].plot(x_iters, y_fro_err_list, "r")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("$|T_2 x - y|_F$")
    axes[1].set_title(r"Jacobi iteration with $y$")

    fig.tight_layout()
    fig.savefig("./p3_y.png")
