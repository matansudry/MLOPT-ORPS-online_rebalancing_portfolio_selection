from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def matrix_projsplx(A,y):
    I = np.eye(n, dtype=np.float32)
    x_t = I[0]
    max_iter = int(T / 10)
    for t in range(1, max_iter):
        step = 2 / (1 + t)
        grad = 2 * np.dot(A, x_t - y)
        min_idx = np.argmin(grad)
        y_t = I[min_idx]
        x_t = x_t + step * (y_t - x_t)
    return x_t

def projsplx(y):
    """function was taken from: https://github.com/swyoon/projsplx"""
    """projsplx projects a vector to a simplex
    by the algorithm presented in
    (Chen an Ye, "Projection Onto A Simplex", 2011)"""
    assert len(y.shape) == 1
    N = y.shape[0]
    y_flipsort = np.flipud(np.sort(y))
    cumsum = np.cumsum(y_flipsort)
    t = (cumsum - 1) / np.arange(1, N + 1).astype('float')
    t_iter = t[:-1]
    t_last = t[-1]
    y_iter = y_flipsort[1:]
    if np.all((t_iter - y_iter) < 0):
        t_hat = t_last
    else:
        # find i such that t>=y
        eq_idx = np.searchsorted(t_iter - y_iter, 0, side='left')
        t_hat = t_iter[eq_idx]
    x = y - t_hat
    # there may be a numerical error such that the constraints are not exactly met.
    x[x < 0.] = 0.
    x[x > 1.] = 1.
    assert np.abs(x.sum() - 1.) <= 1e-5
    assert np.all(x >= 0) and np.all(x <= 1.)
    return x

def f(x,t):
    r = R[t]
    return -np.log(x@r)

def gradient_f(x,t):
    a = -1/np.dot( R[t-1],x)
    a = np.dot(a,R[t-1])
    return a

def get_r_t(t):
    price_end_t = A[:,t+1]
    price_start_t = A[:,t]
    r_t = price_end_t/price_start_t
    r_t_short = price_start_t / price_end_t
    all_assets = np.concatenate((r_t,r_t_short))
    return all_assets

def compute_wealth(w,x,t):
    r = R[t]
    return w*(np.dot(x,r))

def show_plot(OGD_scores,OEG_scores,ONS_scores,best_stock,best_portfolio_scores):
    plt.plot(range(1,T+1), OGD_scores,'red')
    plt.plot(range(1,T+1), OEG_scores,'blue')
    plt.plot(range(1,T+1), ONS_scores,'black')
    plt.plot(range(1,T+1), best_stock,'orange')
    plt.plot(range(1,T+1), best_portfolio_scores,'green')
    plt.legend(['OGD','OEG','ONS','Best Fixed Stock','Best Fixed Rebalancing Portfolio'], loc='best')
    plt.yscale('log')
    # plt.xscale('log')
    plt.grid(True)
    plt.ylabel('Total Wealth')
    plt.xlabel('#rounds')
    plt.title('Portfolios Wealth Through Time')
    plt.savefig('Portfolios.png')
    plt.show()
    plt.clf()

def run_OGD(step):
    x1 = np.random.uniform(0,1,n)
    x_t = x1/sum(x1) #normalize to get unit simplex
    wealth = np.zeros(T)
    wealth[0] = compute_wealth(1,x_t,0)
    for t in range(1,T):
        x_t = x_t - step*gradient_f(x_t,t)
        x_t = projsplx(x_t) #projection onto simplex
        wealth[t] = compute_wealth(wealth[t-1],x_t,t)
    return wealth

def run_OEG(step):
    x_t = np.empty(n)
    x_t.fill(1/n)
    wealth = np.zeros(T)
    wealth[0] = compute_wealth(1,x_t,0)
    for t in range(1,T):
        g_t = gradient_f(x_t,t)
        x_t_plus1 = np.asarray([xi*np.exp(-step*g_t[i]) for i,xi in enumerate(x_t)])
        x_t = x_t_plus1 / sum(x_t_plus1)
        wealth[t] = compute_wealth(wealth[t-1],x_t,t)
    return wealth

def run_ONS(epsilon,gamma):
    x1 = np.random.uniform(0,1,n)
    x_t = x1/sum(x1) #normalize to get randomized unit simplex
    A = np.empty((n,n))
    np.fill_diagonal(A,epsilon)
    wealth = np.zeros(T)
    wealth[0] = compute_wealth(1,x_t,0)
    for t in range(1,T):
        g_t = gradient_f(x_t,t)
        V = np.outer(g_t,g_t)
        A = A + V
        y_t_plus1 = x_t - (1/gamma)*np.dot(np.linalg.inv(A),g_t)
        x_t = matrix_projsplx(A,y_t_plus1) #projection onto simplex
        wealth[t] = compute_wealth(wealth[t-1],x_t,t)
    return wealth

def get_best_stock_scores():
    idx_best_single = np.argmax(np.prod(R, axis=0))
    x_t = np.zeros(n)
    x_t[idx_best_single] = 1
    wealth = np.zeros(T)
    wealth[0] = compute_wealth(1,x_t,0)
    for t in range(1,T):
        wealth[t] = compute_wealth(wealth[t-1],x_t,t)
    return wealth

def get_best_portfolio_scores():
    x_t = get_opt_vec()
    wealth = np.zeros(T)
    wealth[0] = compute_wealth(1,x_t,0)
    for t in range(1,T):
        wealth[t] = compute_wealth(wealth[t-1],x_t,t)
    return wealth

def full_gradient(x):
    a = -1/np.dot( R,x)
    a = np.dot(a,R)
    return a

def get_opt_vec():
    I = np.eye(n, dtype=np.float32)
    x_t = np.empty(n)
    x_t.fill(1/n)
    max_iter = int(T/10)
    for t in range(1,max_iter):
        step = 2 / (1 + t)
        R_grad = full_gradient(x_t)
        min_idx = np.argmin(R_grad)
        y_t = I[min_idx]
        x_t = (1-step)*x_t + step*y_t
    return x_t

def get_wealth_scores():
    global R
    R = np.empty_like(A)
    R = np.delete(R, -1, axis=1)
    R = np.concatenate((R, R))
    for t in range(T):
        R[:, t] = get_r_t(t)
    R = np.transpose(R)
    best_portfolio_scores = get_best_portfolio_scores()
    best_stock_scores = get_best_stock_scores()
    D = np.sqrt(2)
    G =  (np.max(np.linalg.norm(R, axis=1) / np.sum(R, axis=1)))
    step = D/(G*np.sqrt(T))
    OGD_scores = run_OGD(step)
    step = np.log(n)/(G*np.sqrt(2*T))
    OEG_scores = run_OEG(step)
    gamma = 0.5*min(1/(4*G*D),1)
    epsilon = 1/((gamma**2)*(D**2))
    ONS_scores = run_ONS(epsilon, gamma)
    return OGD_scores,OEG_scores,ONS_scores,best_stock_scores,best_portfolio_scores

data = loadmat('data_490_1000.mat')
A = data['A']
T = A.shape[1]-1
n = 2*A.shape[0]

if __name__ == "__main__":

    OGD_scores, OEG_scores, ONS_scores, best_stock_scores, best_portfolio_scores = get_wealth_scores()
    show_plot(OGD_scores,OEG_scores,ONS_scores,best_stock_scores,best_portfolio_scores)