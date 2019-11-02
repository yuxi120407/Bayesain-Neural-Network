# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:05:28 2019

@author: Xi Yu
"""

import tensorflow as tf
import numpy as np
import pandas as pd
#import tensorflow.contrib.eager as tfe

print(tf.__version__)
print(tf.__git_version__)

tf.compat.v1.enable_eager_execution()

#%%

# set data dimensions
K = 3
V = 5
D = 100
N = 100

# set the seed
np.random.seed(2014)

# beta prior parameters
eta = np.ones(V) * 1e-1

# beta profiles
beta = np.random.dirichlet(alpha=eta, size=K)

# theta prior parameters
alpha = np.ones(K) * 1e-1
# alpha[0] = 10

# document's prior topic allocation
theta = np.random.dirichlet(alpha=alpha, size=D)

# word's topic membership
z = [np.random.choice(K, size=N, replace=True, p=theta[d, :]) for d in range(D)]
z = np.vstack(z)

# actual words and counts
w = [np.array([np.random.choice(V, size=1, p=beta[k,:])[0] for k in z[d, :]]  + list(range(V))) for d in range(D)]
nw = [np.unique(w[d], return_counts=True)[1] for d in range(D)]
nw = np.vstack(nw)
w = np.vstack(w)

nw = tf.convert_to_tensor(nw, dtype=tf.float32)
nw = tf.Variable(initial_value=tf.transpose(nw),
                 name="nw_vd")
#%%
print("beta:")
pd.DataFrame(np.round(np.transpose(beta), decimals=3))
#%%
print("theta:")
pd.DataFrame(np.round(theta, decimals=3)).head(6)
#%%
print("documents word counts:")
pd.DataFrame(tf.transpose(nw).numpy()).head(6)
#%%
# initialize LDA parameters
def initialize_variables(K, V, D, alpha=1e-1, eta=1e-1, seed=2014):
    """
    Initialize parameters of LDA model returning adequate Tensors.

    args:
    
        K (int): number of LDA components 
        V (int): vocabulary size
        D (int): number of documents
        alpha (float): hyperparameter for theta prior
        eta (float): hyperparameter for beta prior
       
       
    returns:
    
        eta: [V] tensor with prior parameters (alpha) for beta
        lambda: [K, V] tensor with posterior word distribution per class
        phi: [K, V, D] tensor with vocabulary membership per document
        gamma: [K, D] tensor
        
    """
    tf.random.set_seed(seed)
    eta = tf.Variable(initial_value=tf.ones(V) * eta, 
                       name="eta_v")
    alpha = tf.Variable(initial_value=tf.ones(K) * alpha, 
                         name="alpha_k")    
    lam = tf.Variable(tf.abs(tf.random.normal(shape=(K, V))), 
                       name="lambda_kv")
    
    phi = tf.Variable(initial_value=tf.random.normal(shape=(K, V, D)), 
                       name="phi_kvd")
    phi.assign(value=tf.nn.softmax(phi, axis=0))
    
    gamma = tf.Variable(initial_value=tf.abs(tf.random.normal(shape=(K, D))), 
                        name="gamma_kd")
    
    e_log_beta = tf.Variable(initial_value=tf.abs(tf.random.normal(shape=(K, V, D))) * .0, 
                        name="e_log_beta_kvd")
    
    e_log_theta = tf.Variable(initial_value=tf.abs(tf.random.normal(shape=(K, V, D))) * .0, 
                        name="e_log_theta_kvd")
    
    return eta, alpha, lam, phi, gamma, e_log_beta, e_log_theta
#%%
# test
eta, alpha, lam, phi, gamma, e_log_beta, e_log_theta = initialize_variables(K, V, D)

#%%
def update_lambda(lam, eta, phi, nw):
    
    K = lam.shape.as_list()[0]
    num_k = lam.shape.as_list()[1]
    for k in range(K):
        lam.assign(tf.tensor_scatter_nd_update(lam, 
                  indices=tf.constant([[k,i] for i in range(num_k)]), 
                  updates=tf.reduce_sum(tf.multiply(phi[k], nw), axis=1) + eta))
        
    return lam
#%%

update_lambda(lam, eta, phi, nw)
print(lam)
#%%
# gamma update
def update_gamma(gamma, alpha, phi, nw):
    
    K = gamma.shape.as_list()[0]
    num_k = gamma.shape.as_list()[1]
    for k in range(K):
        gamma.assign(tf.tensor_scatter_nd_update(gamma, 
                  indices=tf.constant([[k,i] for i in range(num_k)]), 
                  updates=tf.reduce_sum(tf.multiply(phi[k], nw), axis=0) + alpha[k]))

        
    return gamma
tmp = gamma.value()
update_gamma(gamma, alpha, phi, nw)
print(gamma)
#%%
def update_e_log_beta(e_log_beta, lam):
    
    K = lam.shape.as_list()[0]
    num_k = lam.shape.as_list()[1]
    for k in range(K):
        e_log_beta.assign(tf.tensor_scatter_nd_update(e_log_beta,
                  indices=tf.constant([[k,i] for i in range(num_k)]),
                  updates=tf.tile(tf.expand_dims(tf.math.digamma(lam[k]) - tf.math.digamma(tf.reduce_sum(lam[k])), axis=1), multiples=[1, D])))
    
    return e_log_beta

print(e_log_beta)
update_e_log_beta(e_log_beta, lam);
print(e_log_beta)
#%%
def update_e_log_theta(e_log_theta, gamma):
    
    e_log_theta.assign(value=tf.tile(tf.expand_dims(tf.math.digamma(gamma) - 
                                                    tf.math.digamma(tf.reduce_sum(gamma, axis=0)), axis=1), multiples=[1, V, 1]))

    return e_log_theta

update_e_log_theta(e_log_theta, gamma)
#%%

import time
start = time.time()

def update_phi(e_log_beta, e_log_theta):
    phi.assign(value=e_log_beta + e_log_theta)
    phi.assign(value=tf.nn.softmax(logits=phi, axis=0))
    return phi


update_phi(e_log_beta, e_log_theta)

end = time.time()
print(end - start)
print(phi)

#%%
nw_kvd = tf.tile(tf.expand_dims(nw / tf.reduce_sum(nw), axis=0), 
                 multiples=[K, 1, 1])
nw_kvd

#%%
def elbo(phi, e_log_beta, e_log_theta, nw_kvd):

    A = tf.reduce_sum(nw_kvd * phi * (e_log_beta + e_log_theta - tf.math.log(phi + 1e-6)))
    return A.numpy()

elbo(phi, e_log_beta, e_log_theta, nw_kvd)

#%%
seed = 1
seed += 1
eta, alpha, lam, phi, gamma, e_log_beta, e_log_theta = initialize_variables(K, V, D)

prev_elbo = 0.0
next_elbo = 0.0
iter = 0

for i in range(100000):
    
    for j in range(100000):
        # E-Step:
        update_e_log_beta(e_log_beta, lam);
        update_e_log_theta(e_log_theta, gamma);
        update_phi(e_log_theta=e_log_theta, e_log_beta=e_log_beta)
        gamma_prev = gamma.value()
        update_gamma(gamma, alpha, phi, nw)
        
        diff = tf.reduce_mean(tf.abs(gamma_prev - gamma.value()))
        if diff < 1e-6:
            break
    
    # M-Step:
    update_lambda(lam, eta, phi, nw)
    
    
    next_elbo = elbo(phi, e_log_beta, e_log_theta, nw_kvd)
#     next_elbo = 0.0
    print("Iteration:", iter, "ELBO:", next_elbo)
    
    diff = np.abs(next_elbo - prev_elbo)
    if diff < 1e-6:
        print("Converged!")
        break
    else:
        iter += 1
        prev_elbo = next_elbo

