# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:48:17 2022

@author: uanjum
"""

import pandas as pd
import numpy as np
import random
import math
import zipfile
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from sklearn import preprocessing

from filterpy.kalman import unscented_transform, JulierSigmaPoints, MerweScaledSigmaPoints
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.kalman import UnscentedKalmanFilter as UKF 
from filterpy.kalman import KalmanFilter 
from numpy.random import multivariate_normal
from filterpy.common import Q_discrete_white_noise
from numpy.random import randn
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# State Transition Function
def f_cv(x, dt):
    '''
    State Transition Function (F): Represents x(t) = Fx(t-1)    
    Parameters
    ----------
    x : input numpy array of size (M, )
    dt : input parameter for MxN matrix F (time difference)

    Returns
    -------
     : output numpy array of size (M, )
    '''
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    return F @ x

# Measurement Function
def h_cv(x):
    '''
    Measurement Function (H): Represents z(t) = Hx(t); Reduces dimenstion of x    
    Parameters
    ----------
    x : input numpy array of size (M, )

    Returns
    -------
     : output numpy array of size (N, )

    ''' 
    H = np.array([[1,  0, 0,  0],
                  [0,  0, 1,  0]])
    # H = np.array([[1,  0, 0,  0],
    #               [0,  1, 0,  0],
    #               [0,  0, 1,  0],
    #               [0,  0, 0,  1]])
    return H @ x

def hx(x):
    return np.array([x[0]])

# Function that applies sigma to State Transition Function
def sigmas_state_function(sigmas, dim_x, delt):
    '''
    Apply F to sigma    
    Parameters
    ----------
    sigmas : input numpy array of size (L, M) where L = number of sigmas and M = dim_x
    dim_x : input dimension = number of sensors
    dt : input parameter for MxN matrix F (time difference)
    
    Returns
    -------
    Fsigmas : output numpy array of size (L, M)

    ''' 
    Fsigmas = np.zeros((sigmas.shape[0], dim_x))
    for i, s in enumerate(sigmas):
        Fsigmas[i] = f_cv(s, delt)
    return Fsigmas

# Function that applies sigma to Measurement Function
def sigmas_measurement_function(sigmas, dim_z):
    '''
    Apply H to sigma    
    Parameters
    ----------
    sigmas : input numpy array of size (L, M) where L = number of sigmas and M = dim_x
    dim_z : size of hidden dimension
    
    Returns
    -------
    Hsigmas : output numpy array of size (L, N)

    ''' 
    Hsigmas = np.zeros((sigmas.shape[0], dim_z))
    for i, s in enumerate(sigmas):
        Hsigmas[i] = h_cv(s)
    return Hsigmas

def auto_correlation(sigmas, musigmas, R):
    '''
    Calculating the autocorrelation   
    Parameters
    ----------
    sigmas : input numpy array of size (L, M) where L = number of sigmas and M = dim_x
    musigmas : numpy array of mean of sigma points of size (M, )
    R : Noise covariance for state measurements
    
    Returns
    -------
    Pzz : covariance matrix of size (M, M)

    '''    
    Pzz = 0
    for sigma in sigmas:
        s = sigma - musigmas
        Pzz += np.outer(s, s)
    Pzz = Pzz / sigmas.shape[0] + R
    return Pzz

def cross_correlation(Fsigmas, Hsigmas, muFsigmas, muHsigmas):
    '''
    Calculating the crosscorrelation   
    Parameters
    ----------
    Fsigmas : input numpy array of size (L, M) where L = number of sigmas and M = dim_x
    muFsigmas : numpy array of mean of sigma points of size (M, )
    Hsigmas : input numpy array of size (L, N) where L = number of sigmas and N = dim_z
    muHsigmas : numpy array of mean of sigma points of size (N, )
    
    Returns
    -------
    Pxz : covariance matrix of size (M, M)

    '''  
    Pxz = 0
    #Pxz = np.zeros((sigmas_f.shape[1],sigmas_h.shape[1]))
    for i in range(Hsigmas.shape[0]):
        Pxz += np.outer(np.subtract(Fsigmas[i], muFsigmas),np.subtract(Hsigmas[i], muHsigmas))
    Pxz /= (Hsigmas.shape[0] - 1)
    return Pxz

def error_distribution(mu, cov, N):
    '''
    Error value to be added to sigma: multivariate normal distribution  
    Parameters
    ----------
    mu : error mean vector of size (M, )
    cov : error covariance matrix of size (M, M)
    N: number of error values of size (N, M)
    
    Returns
    -------
    error : error vector

    '''  
    error = multivariate_normal(mu, cov, N)
    return error

def sigmas_with_error(sigmas, error):
    '''
    Adding Error to sigmas  
    Parameters
    ----------
    Fsigmas : input numpy array of size (L, M) where L = number of sigmas and M = dim_x
    muFsigmas : numpy array of mean of sigma points of size (M, )
    Hsigmas : input numpy array of size (L, N) where L = number of sigmas and N = dim_z
    muHsigmas : numpy array of mean of sigma points of size (N, )
    
    Returns
    -------
    Pxz : covariance matrix of size (M, M)

    '''  
    sigmas += error
    return sigmas

def rmse(x, x_hat):
    '''
    Root mean square error   
    Parameters
    ----------
    x : actual values
    x_hat : estimated values
    
    Returns
    -------
    rmse_err : rmse value

    '''  
    rmse_err = math.sqrt(mean_squared_error(x, x_hat)) 
    return rmse_err

def normalize_data(input):
    '''
    Normalize Input Data   
    Parameters
    ----------
    x : input numpy matrix
    
    Returns
    -------
    input_normalized : normalized output

    '''  
    # using sklean to normalize data using normalization formula is (x - x_min)/(x_max - x_min)
    input_min = np.min(zs, axis=0)   
    input_max = np.max(zs, axis=0)
    input_normalized = (input - input_min) / (input_max - input_min)
    return input_normalized

dt = 1
dim_x, dim_z = 4, 2 # number of input sensors, dim_z is the number of hidden states
std_x, std_y = .3, .3

# Initial mean and covariance matrix for UKF
x = np.array([0., 0., 0., 0.])
P = np.eye(dim_x) * 100.

Q = np.eye(dim_x)
Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)
#R = np.diag([std_x**2, std_y**2, std_x, std_y])
R = np.diag([std_x**2, std_y**2])
N = 2*dim_x + 1
# Points for Unscented Transform for UKF
points = JulierSigmaPoints(n=dim_x,kappa=3-dim_x)
xinit = np.zeros(dim_x)
Pinit = np.eye(dim_x)
# Points for Normal distribution for EKF
sigmas1 = multivariate_normal(mean=x, cov=P, size=N)

ekf_X = []
ukf_X = []
sigmas = sigmas1

# Input simulated data
zs = [np.array([i + randn()*std_x, math.sin(i)**2 + randn()*0.3]) for i in range(100)] 
zs[80:90] = [np.array([i + randn()*std_x, math.sin(i/5) + randn()*0.3]) for i in range(80,90,1)]
zs[30:40] = [np.array([i + randn()*std_x, math.sin(i/5) + randn()*0.3]) for i in range(30,40,1)] 
# zs = [np.array([i, math.sin(i)**2]) for i in range(100)] 
# zs[80:90] = [np.array([i, math.sin(i/5)]) for i in range(80,90,1)]
# zs[30:40] = [np.array([i, math.sin(i/5)]) for i in range(30,40,1)]   
zs0 = np.array(zs) # input to EKF
zs1 = normalize_data(np.array(zs)) # input to UKF
# plt.plot(zs0[:,0], zs0[:,1], color='black', label='inp')

for j in range(len(zs)):
    # Predict Ensemble Kalman Filter
    Fsigmas = sigmas_state_function(sigmas, dim_x, dt)
    err_predict = error_distribution(np.zeros(dim_x), Q, N)
    Fsigmas = sigmas_with_error(Fsigmas, err_predict)
    P_ekf = auto_correlation(sigmas, x, Q)
    # Predict Unscented Kalman Filter
    sigmas0 = points.sigma_points(xinit, Pinit)
    Fsigmas0 =  sigmas_state_function(sigmas0, dim_x, dt)
    z_hat, P_hat = unscented_transform(Fsigmas0, points.Wm, points.Wc, Q)
        
    # Update Ensemble Kalman Filter
    sigmas_h_ekf = sigmas_measurement_function(Fsigmas, dim_z)
    z_mean = np.mean(sigmas_h_ekf, axis=0)
    P_zz_ekf = auto_correlation(sigmas_h_ekf, z_mean, R)
    P_xz_ekf = cross_correlation(Fsigmas, sigmas_h_ekf, x, z_mean)
    ekfK = np.dot(P_xz_ekf, np.linalg.inv(P_zz_ekf))  
    err_update = error_distribution(np.zeros(dim_z), R, N)
    sigmas_ekf = Fsigmas
    for i in range(N):
        sigmas_ekf[i] += np.dot(ekfK, zs0[j] + err_update[i] - sigmas_h_ekf[i])
    x_temp = np.mean(sigmas_ekf, axis=0)
    P_temp = P - np.dot(np.dot(ekfK, P_zz_ekf), ekfK.T)
    
    sigmas = sigmas_ekf
    x = x_temp
    P = P_temp
    ekf_X.append(x_temp)
    
    # Update Unscented Kalman Filter
    sigmas_h_ukf =  sigmas_measurement_function(Fsigmas0, dim_z)
    zx_hat, Px_hat = unscented_transform(sigmas_h_ukf, points.Wm, points.Wc, R)          
    P_xz_ukf = cross_correlation(Fsigmas0, sigmas_h_ukf, z_hat, zx_hat)           
    ukfK = np.dot(P_xz_ukf, np.linalg.inv(Px_hat))
        
    x_new = z_hat + np.dot(ukfK, zs1[j] - zx_hat)
    P_new = P_hat - np.dot(ukfK, np.dot(Px_hat, ukfK.T))
    xinit = x_new
    Pinit = P_new
    ukf_X.append(x_new)
    
ukf_X = np.array(ukf_X)
ekf_X = np.asarray(ekf_X)
rmse_ekf = round(rmse(zs0, np.array((ekf_X[:,0], ekf_X[:,2])).T),4)
rmse_ukf = round(rmse(zs1, np.array((ukf_X[:,0], ukf_X[:,2])).T),4)

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.plot(ekf_X[:,0], ekf_X[:,2], color='blue', label='EKF');
plt.plot(zs0[:,0], zs0[:,1], color='black', label='inp');
plt.title("EnF: RMSE=%f" % rmse_ekf)

plt.subplot(1, 2, 2) # row 1, col 2 index 1
plt.plot(ukf_X[:,0], ukf_X[:,2], color='red', label='EKF');
plt.plot(zs1[:,0], zs1[:,1], color='black', label='inp');
plt.title("UKF: RMSE=%f" % rmse_ukf)

plt.show()

print(rmse(zs0, np.array((ekf_X[:,0], ekf_X[:,2])).T))
print(rmse(zs1, np.array((ukf_X[:,0], ukf_X[:,2])).T))



