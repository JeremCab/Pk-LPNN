# ========== #
# Librairies #
# ========== #

# install the following librairies if needed

# !pip install scikit-learn
# !pip install scipy
# !pip install time-python
# !pip install numdifftools

import numpy as np
import pandas as pd
import numdifftools as nd
import time as gettime
import time
from scipy.integrate import odeint, solve_ivp

from sklearn.linear_model import Lasso

from .accelerated_proximalGD import *

# import matplotlib.pyplot as plt


# ======= #
# Pk-LPNN #
# ======= #

# # Examples of parameters
# p, n, Nz, k, sigma = 512, 128, 25, 5000, 0.02
# mu_0 = 0.5

def initialization(p, n, Nz, k, sigma):
    """
    Doc
    ...
    """
    
    # 1. Define beta
    mask = np.random.choice(p, p - Nz, replace=False)
    beta = np.random.randint(0, 2, size=(p))
    beta = 2*beta - 1
    beta[mask] = 0
    
    eta_prime = Nz - ((p-Nz)/k)
    alpha = np.sqrt( np.square(eta_prime/Nz) - np.square(1/k) )
    beta = alpha * beta
    
    # 2. Define X
    X = np.random.randn(n, p)
    X = np.where(X < 0, -1, 1)
    X = X / np.linalg.norm(X, axis=0) # normalize
    
    # 3. Define Y
    Y = np.matmul(X, beta) + np.random.normal(0, sigma, size=(n))
    
    return beta, X, Y


def T_kappa(u, kappa=1):
    """Thresholding activation function for LASSO-LPNN"""
    
    mask = np.array(np.abs(u) > kappa, dtype=np.int8)
    
    return mask * (u -  kappa*np.sign(u))


def k1_norm(beta, k, method="Pk-LPNN_v2"):
    """k1_norm"""
    
    if method == "Pk-LPNN_v1":
        
        # note that for too large k, np.cosh(k * beta) explodes, cannot be computed... 
        norm = np.sum( np.log(2*(1 + np.cosh(k * beta))) ) / k
        
    elif method == "Pk-LPNN_v2":
        
        norm = np.sum( np.sqrt(np.square(beta) + 1/np.square(k)) )
        
    return norm


def beta_0_and_mu_0(p, Nz, k, approach=3, amplitude=1, 
                    sigma_1=0.03, ratio_beta0=0.95,
                    mu_0=None, method="Pk-LPNN_v2"):
    """Initialization of beta_0 and mu_0."""
    
    eta = Nz
    
    if method == "LASSO-LPNN":
        
        approach = 1
    
    elif method == "Pk-LPNN_v1":
        
        approach = 2
        
    elif method == "Pk-LPNN_v2":
        
        approach = 3
    
    if approach == 1:

        beta_0 = amplitude * sigma_1 * np.random.normal(0, 1, size=(p))

    elif approach == 2:

        beta_g = amplitude * np.random.normal(0, 1, size=(p))
        beta_2 = (0.95 * eta) / k1_norm(beta_g, k, method=method)
        beta_0 = beta_2 * beta_g
        
    elif approach == 3:
        
        alphax0 = np.sqrt( ((ratio_beta0 * eta)/p)**2 - 1 / k**2)
        beta_0 = alphax0 * amplitude * np.sign(np.random.normal(size=(p,)))
    
    return beta_0, mu_0


# # test
# beta_0, mu_0 = beta_0_and_mu_0(p=p, Nz=Nz, k=k, mu_0=mu_0, method="Pk-LPNN_v2")
# beta_0.shape, mu_0

# beta, X, Y = initialization(p, n, Nz, k, sigma)
# beta.shape, X.shape, Y.shape


class ConditionError(Exception):
    pass


def check_conditions(beta, X, beta_0, n, Nz, k, verbose=False, method="Pk-LPNN_v2"):
    """
    Check conditions for $P_k-LPNN$ problem.
    """
    
    eta = Nz
    
    # 1. LASSO contraints
    if np.count_nonzero(beta) == Nz:
        if verbose:
            print("OK: nb components ≠ 0 equal to Nz")
    else:
        raise ConditionError("ERROR: nb components ≠ 0 not equal to Nz")
        
    norm_beta = np.linalg.norm(beta, ord=1)
    
    if norm_beta <= eta:
        if verbose:
            print(f"OK: ||beta||_1 = {norm_beta} ≤ eta")
    else:
        raise ConditionError(f"ERROR: ||beta||_1 = {norm_beta} > eta")

    # 2. P_k constraint
    if method == "Pk-LPNN_v2": # no need to check this for Pk-LPNN_v1
        
        norm_beta = k1_norm(beta, k, method=method)

        if np.round(norm_beta, decimals=5) <= eta:
            if verbose:
                print(f"OK: ||beta||_k,1 = {norm_beta} ≤ eta")
        else:
            raise ConditionError(f"ERROR: ||beta||_k,1 = {norm_beta} > eta")
    
    # 3. rank of X
    if np.linalg.matrix_rank(X) == n:
        if verbose:
            print("OK: X is full rank")
    else:
        raise ConditionError("X is not full rank")
        
    # 4. check beta_0
    if not method == "LASSO-LPNN": # no need to check this for LASSO-LPNN
        
        norm_beta = k1_norm(beta_0, k, method=method)

        if norm_beta <= eta:
            if verbose:
                print(f"OK: ||beta_0||_k,1 = {norm_beta} ≤ eta")
        else:
            raise ConditionError(f"ERROR: ||beta_0||_k,1 = {norm_beta} > eta")


# # test 
# check_conditions(beta, X, beta_0, n, Nz, k, method="Pk-LPNN_v2")

def gradient_norm_k1(beta, k, method="Pk-LPNN_v2"):
    """gradient of beta. Depends on the method."""
    
    if method == "Pk-LPNN_v1":
        
        grad = np.sinh(k * beta) / (1 + np.cosh(k * beta))
    
    elif method == "Pk-LPNN_v2":
        
        grad = beta / np.sqrt( np.square(beta) + np.square(1/k) )
    
    return grad


# def LASSO-LPNN(t, z, X, Y, Nz):
#     """
#     Define LASOO-LPNN dynamical system
#     """
    
#     eta = Nz
    
#     # We have u = z[:-1] and mu = z[-1]
#     db = np.matmul(X.T, (Y - np.matmul(X, T_kappa(z[:-1]))))
        
#     du_dt = db - z[-1] * (z[:-1] - T_kappa(z[:-1]))    # Variable neurons (p)
#     dmu_dt = np.linalg.norm(T_kappa(z[:-1]), 1) - eta  # Lagrange neuron  (1)
#     dz_dt = np.hstack([du_dt, dmu_dt])

#     return dz_dt


def LPNN(t, z, X, Y, Nz, k, method="Pk-LPNN_v2", pbar=None):
    """
    Define P_k-LPNN dynamical system
    """
        
    eta = Nz
    
    if method == "LASSO-LPNN":
        
        db = np.matmul(X.T, (Y - np.matmul(X, T_kappa(z[:-1]))))
        du_dt = db - z[-1] * (z[:-1] - T_kappa(z[:-1]))    # Variable neurons (p)
        dmu_dt = np.linalg.norm(T_kappa(z[:-1]), 1) - eta  # Lagrange neuron  (1)
        dz_dt = np.hstack([du_dt, dmu_dt])
        
        #time.sleep(0.001)
        if pbar != None:
            pbar.update()

    else:
                
        db = np.matmul(X.T, (Y - np.matmul(X, z[:-1])))        
        dbeta_dt = db - z[-1] * gradient_norm_k1(z[:-1], k=k, method=method) # Variable neurons (p)
        dmu_dt = k1_norm(z[:-1], k=k, method=method) - eta                   # Lagrange neuron  (1)
        dz_dt = np.hstack([dbeta_dt, dmu_dt])
        
        #time.sleep(0.001)
        if pbar != None:
            pbar.update()

    return dz_dt


def experiment(p, n, Nz, k, sigma, beta_0, mu_0, method="Pk-LPNN_v2"):
    """
    Launch an experiment with the given parameters for simulated data.
    ...
    """
    
    # 1. define variables
    beta, X, Y = initialization(p, n, Nz, k, sigma)
    eta = Nz
    beta_0, mu_0 = beta_0_and_mu_0(p=p, Nz=Nz, k=k, mu_0=mu_0, method=method)
    
    # 2. check conditions
    check_conditions(beta, X, beta_0, n, Nz, k, method=method)
    
    # 3. solve P_k-LPNN problem
    z0 = np.hstack([beta_0, mu_0])
    t_span = (0, 150)
    t = t_span[1]
    
    start_time = gettime.time() # t0
    sol = solve_ivp(LPNN, t_span=t_span, y0=z0, args=(X, Y, eta, k, method), 
                    method="RK45", dense_output=False, max_step=0.1, atol=1.2e-4, rtol=1e-4)
    end_time = gettime.time()  # t1

    time = end_time - start_time
    
    return {"solution": sol, 
            "time": time, 
            "beta": beta, 
            "X": X,
            "Y": Y,
            "params (p, n, Nz, sigma, beta_0, mu_0)": [p, n, Nz, sigma, beta_0, mu_0]}


# # test 
# solution_d = experiment(p, n, Nz, sigma, beta_0, mu_0)


# ===== #
# LASSO #
# ===== #


def LASSO_skl(X, Y, alphas=np.logspace(-3, 3, num=10)):
    """
    Fit a LASSO (scikit-learn) on X, Y for different alphas 
    and stores results in a dict.
    """
    
    sol_l = []
    
    for alpha in alphas:
        
        # print("alpha:", alpha)
        
        lasso = Lasso(alpha=alpha)
        lasso.fit(X=X, y=Y)
        beta_hat = lasso.coef_
        sol_l.append(beta_hat)
        
    return sol_l


# ========================= #
# Proximal Gradient Descent #
# ========================= #


def ProximalGD(X, Y, 
               alphas=np.linspace(0.5, 1.5, num=5), 
               lamdas=np.linspace(0.3, 0.9, num=5)):
    """Proximal gradient descent."""
    
    p = X.shape[1]
    
    sol_l = []
    
    for alpha in alphas:
        
        for lamda in lamdas:
            
            # print("alpha:", alpha, "\t", "lamda:", lamda)
        
            max_iter = 1000

            Y_ = Y.reshape(-1, 1)          # for code compatibility
            beta_k = np.random.rand(p, 1)  # initialization

            for i in range(max_iter):

                # Gammak = 0.01
                Gammak = 1 / np.linalg.norm(X.T.dot(X))      

                while True:

                    beta_kplus1 = beta_k - Gammak * grf(X, beta_k, Y_)  # GD step

                    if f(X, beta_kplus1, Y_) <= m(beta_kplus1, beta_k, X, Y_, Gammak):
                        break

                    else:
                        Gammak = lamda * Gammak

                beta_kplus1 = proxNorm1(beta_kplus1, Gammak * alpha) # proximal operation

                obj_diff = np.linalg.norm(obj(X, beta_kplus1, Y_, alpha) - obj(X, beta_k, Y_, alpha))

                if(obj_diff < 1e-6):
                    break

                beta_k = beta_kplus1 # update beta_k

            beta_k = beta_k.reshape(-1,)
            
            sol_l.append(beta_k)
    
    return sol_l