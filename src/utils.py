import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import zero_one_loss


def threshold_soft(beta_sol, mu_sol):
    """Soft-thresholding function"""
    
    mask = np.array(np.abs(beta_sol) > mu_sol, dtype=np.int8)
    
    return mask * (beta_sol - mu_sol * np.sign(beta_sol))


def threshold_soft_new(beta_sol, mu_sol):
    """Soft-thresholding function"""
    
    c_max = beta_sol[beta_sol > 0].max()
    c_min = beta_sol[beta_sol > 0].min()
    thresold_new = (c_max + c_min) / 2
    
    mask = np.array(np.abs(beta_sol) > thresold_new, dtype=np.int8)
    
    return mask * (beta_sol - mu_sol * np.sign(beta_sol))


def threshold(beta_sol, beta, threshold=0.5):
    """Thresholding function"""
    
    mask = np.array(np.abs(beta_sol) > threshold, dtype=np.int8)
    
    # return mask * np.sign(beta_sol) * np.max(beta)
    return mask * np.max(beta)


def thresholdTernary(beta_sol, beta, threshold=0.5):
    """Thresholding function"""
    
    mask = np.array(np.abs(beta_sol) > threshold, dtype=np.int8)
    
    return mask * np.sign(beta_sol) * np.max(beta)


def compute_losses(beta_sol_t, beta):
    """MSE, zero_one_loss and macro f1"""
    
    beta_ = np.sign(beta).astype(int)
    beta_sol_t_ = ((beta_sol_t != 0) * np.sign(beta_sol_t)).astype(int)
    
    mse = mean_squared_error(beta, beta_sol_t)
    zero_one = zero_one_loss(beta_, beta_sol_t_)
    f1 = f1_score(beta_, beta_sol_t_, average="macro")
    
    return mse, zero_one, f1


def compute_losses_LASSO(beta_sol_t, beta):
    """
    MSE, zero_one_loss and macro f1 for the LASSO method.
    For each alpha in np.logspace(-3, 1, num=10), 
    we compute the 3 losses of the corresponding reconstructed signal.
    As final losses, we take those associated with the best F1 score.
    """
    
    # beta_ = beta.astype(int)
    
    losses_l = []
    
    for b in beta_sol_t:
        
        mse, zero_one, f1 = compute_losses(b, beta)
        losses_l.append((mse, zero_one, f1))
    
    losses_l.sort(key=lambda x: x[2], reverse=True)
    
    return losses_l[0]


def compute_losses_avg(signals, reconstructed_signals, method="Pk-LPNN_v2"):
    """Compute mean and std losses for an experiment"""

    losses_d = {"mse" : [], "zero_one" : [], "f1" : []}

    for beta, beta_sol_t in zip(signals, reconstructed_signals):
        
        if method == "LASSO-LPNN" or method == "Pk-LPNN_v1" or method == "Pk-LPNN_v2":
            mse, zero_one, f1 = compute_losses(beta_sol_t, beta)
        
        elif method == "LASSO" or method == "PGD":
            mse, zero_one, f1 = compute_losses_LASSO(beta_sol_t, beta)

        losses_d["mse"].append(mse)
        losses_d["zero_one"].append(zero_one)
        losses_d["f1"].append(f1)

    for k, v in losses_d.items():
        mean = np.mean(v)
        std = np.std(v)
        losses_d[k] = (mean, std)

    return losses_d