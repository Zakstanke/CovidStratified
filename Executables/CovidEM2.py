# -*- coding: utf-8 -*-
"""
COVID Estimization-Maximization
Created on Wed Jan 12 19:17:00 2022

@author: zakst
"""

import numpy as np
from numpy import linalg as la
import pandas as pd
from scipy import optimize, stats
from math import log2, log10, isclose
import math
import warnings
import numdifftools as nd
from ci_rvm import find_CI

import os
#os.chdir('C:/Users/zakst/Documents/NIH')
# os.getcwd()


# C:\Users\zakst\AppData\Local\Programs\Python\Python310\Scripts

#####################################################################
# Copied this section from my Markdown file so you can see how I expect the data to look


# =============================================================================
# # Read in data
# cases = pd.read_csv('Data/cases.csv', index_col=0)
# cases = cases.sort_index(ascending = True)
# cases_dict = {}
# cases_mat = cases.to_dict("split")["data"] # temp variable to hold onto arrays
# for i in range(len(cases)):
#     cases_dict[cases.index[i]] = cases_mat[i]
# del i
# del cases_mat
# 
# prem = {
# "can" : pd.read_csv('Data/premCan.csv', index_col=0).to_numpy(),
# "chn" : pd.read_csv('Data/premChn.csv', index_col=0).to_numpy(),
# "gbr" : pd.read_csv('Data/premGbr.csv', index_col=0).to_numpy(),
# "isr" : pd.read_csv('Data/premIsr.csv', index_col=0).to_numpy(),
# "ita" : pd.read_csv('Data/premIta.csv', index_col=0).to_numpy(),
# "nld" : pd.read_csv('Data/premNLD.csv', index_col=0).to_numpy(),
# }
# # to retrieve entries: prem["can"]
# 
# countries = list(prem)
# =============================================================================


#####################################################################


def case_dim_split2(list_in, split = 2, ignore_last = True):
    # Function for splitting up dimensions within case data
    # "split" is the number of groups to be made within each decade
    # If "cutoff" is True, then the last column will be ignored.
    #   This is used if the last column does not represent a defined length of time 
    #   (ex: ages 80 and greater instead of ages 80-89)
    cutoff = 0
    if ignore_last: cutoff = 1
    new_list = [0 for x in range((len(list_in)-cutoff)*split)]
    for i in range(len(list_in)- cutoff):
        for j in range(split):
            # Initial guess for proportions is that cases are evenly split
            new_list[(i * split)+j] = list_in[i]/split 
    assert(int(sum(list_in[0:(len(list_in)-cutoff)])) == int(sum(new_list))),"%s,\t%s" %(sum(list_in[0:(len(list_in)-cutoff)]), sum(new_list))
    return new_list

# Returns Kullback-Leibler Divergence 
def kl_div2(p, q, logbase = math.e):    
    assert(len(p) == len(q)), "Distributions not the same length\nP:(%d)\tQ:(%d)" % (len(p), len(q))
    
    #check for negative values
    neg = False
    for i in range(0,len(p)):
        if (q[i] < 0): neg = True
        if (p[i] < 0): neg = True
    
    if (neg): 
        #print("P: ", p)
        #print("Q: ", q)
        #warnings.warn("Negative number detected")
        return math.inf
    
    if (neg): return math.inf
    
    val = 0
    for i in range(0, len(p)):
        val = val + (p[i] * math.log((p[i]/q[i]), logbase)) # defaults to natural log
    assert(val >= 0), "Calculated KL Divergence is negtive: (%d)" % val
    return val

# Tests known calculation for Kullback-Leibler Divergence
# Just runs through a simple test case
def _test_kl_div2():
    p_test = [0.10, 0.40, 0.50]
    q_test = [0.80, 0.15, 0.05]
    # Check accuracy to 6 decimal places
    temp = kl_div2(p_test, q_test, 2)
    assert (isclose(temp, 1.926979, rel_tol=1e-6)), temp
    temp = kl_div2(q_test, p_test, 2)
    assert (isclose(temp, 2.021647, rel_tol=1e-6)), temp
    return "Test executed successfully"


# Recieves array of 7 (age groups excluding 20-29), outputs diagonal 16x16 matrix
def theta_builder2(array_in, split = 2, diag = True):
    if (len(array_in) != 7): warnings.warn("Dimentions not equal to 7")
    # Insert value for group20-29 here so that everything scales around it
    theta = np.insert(array_in, 2, 1)
    
    # Split decades up
    length = len(theta)
    for i in range(0,length):
         for j in range(0,split-1):
              theta = np.insert(theta, length-1-i, theta[length-1-i])
    
    if (diag): theta = np.diag(theta)
    return theta

# Returns Perron-Frobenius eigenvector from matricies S, C, and V
# Always returns positive elements
# s & v should be arrays of length 7, c should be 16x16 matrix
def scv_eig2(s_in, c_in, v_in, debug = False):
    # Turn s & v into square matricies
    s = theta_builder2(s_in)
    c = c_in
    v = theta_builder2(v_in)
    
    # Ensure that all matricies are the same square dimensions
    assert (len(c) == len(c[0])), "not square"
    assert ((len(s) == len(c)) & (len(c) == len(v))), "rows not equal"
    assert ((len(s[0]) == len(c[0])) & (len(c[0]) == len(v[0]))), "cols not equal"
    builder = np.matmul(np.matmul(s,c),v)
    eigs = la.eig(builder)
    eig = eigs[1][:,0]
    best = -math.inf
    for i in range(0, len(eigs[0])):
        if (eigs[0][i] > best): 
            best = eigs[0][i]
            eig = eigs[1][:,i]
    
    # Ensure all elements have the same sign
    neg = False
    pos = False
    for i in range(0,len(eig)):
        if (eig[i] < 0): neg = True
        if (eig[i] >= 0): pos = True
    assert((neg == True) or (pos == True))
    if ((neg == True) and (pos == True)):
        if (debug):
            print("S: ",s_in)
            print("C: ",c_in)
            print("V: ",v_in)
            print("Eig: ", eig)
        warnings.warn("Eigenvector contains both negative and positive elements")
            
    if (neg): eig = (-1 * eig) # If negative, flip eigvector so all elements are positive
    
    # Scale eigenvector so it sums to 1
    eig_sum = sum(eig)
    eig = [x/eig_sum for x in eig] 
    
    # Remove complex element from floats
    for i in range(0, len(eig)):
        eig[i] = eig[i].real
    
    assert(sum(np.iscomplex(eig)) == 0), eig #Assert real
    #assert(sum((i > 0) for i in eig) == len(eig)), eig #Assert positive
    assert(isclose(sum(eig), 1, rel_tol=1e-6)), sum(eig) #Assert sums to 1
    if (debug):
        print(eig)
        print(type(eig[1]))
    
    return eig


# Per-Country calculation from equation 7 from pickling
# total cases per country times kl divergence: (n_tot * D(u_hat || u_tilda))
# theta0 is a 14 length array, containing two 7-length arrays for s & v
def Covid_KL_k2(theta0, prem_in, kcases_in, debug = False):
    
    for i in range(0, len(theta0)):
        if (theta0[i] < 0): 
            if (debug): 
                print("Detected negative values in theta")
                print(theta0)
            return math.inf
        if (np.isnan(theta0[i])):
            if (debug): 
                print("Detected NaN in theta")
                print(theta0)
            return math.inf
    
    
    total_cases = 0
    kcases = case_dim_split2(kcases_in, ignore_last=True)
    for i in range(0,len(kcases)):
        total_cases = total_cases + kcases[i] #Find total cases per country
    u_hat = [element / total_cases for element in kcases]
    
    u_tilda = scv_eig2(s_in = theta0[0:(int(len(theta0)/2))], c_in = prem_in, v_in = theta0[(int(len(theta0)/2)):])
    # .A1 is needed because theta0[x,:] is returning a nested array for some reason
    assert(isclose(sum(u_tilda), 1, rel_tol=1e-3)), "u-tilda: %s,\tsum: %s" %(u_tilda, sum(u_tilda))
    assert(isclose(sum(u_hat), 1, rel_tol=1e-3)), "u-hat: %s,\tsum: %s" %(u_hat, sum(u_hat))
        
    return kl_div2(u_hat, u_tilda)


def param_trans2(params_in):
    if (type(params_in) == type(np.float64(1.11))):
        return np.exp(params_in)
    params = [0]*len(params_in)
    for i in range(0,len(params_in)):
        params[i] = np.exp(params_in[i])
    return params

# Equation 7 from pickling
# Sums per-country calculations from previous function
# cases_in -> pandas dataframe, prem_in -> dictionary, country_codes -> string array
def Covid_KL2(theta0, prem_in, cases_in, country_codes):
    
    theta0 = param_trans2(theta0)
    
    kl_sum = 0
    #Sum the countries together
    for i in range(0, len(country_codes)):
        # Asserts check that the letter codes can be used on both the cases dataframe and prem dictionary
        # Unlikely the asserts themselves will be triggering, but the call inside the function will throw its own errors if something is wrong
        #assert(prem_in[country_codes[i]].any != None)
        #assert(cases_in[country_codes[i]].any != None)
        res = Covid_KL_k2(theta0, prem_in[country_codes[i]], cases_in[country_codes[i]])
        kl_sum = kl_sum + res
        
    return kl_sum


# global variables and functions for CI calculation
prem_global = {}
cases_global = {}
countries_global = []

def set_prem(prem_in):
    global prem_global
    prem_global = prem_in
def get_prem():
    global prem_global
    return prem_global

def set_cases(cases_in):
    global cases_global
    cases_global = cases_in
def get_cases():
    global cases_global
    return cases_global

def set_countries(countries_in):
    global countries_global
    countries_global = countries_in
def get_countries():
    global countries_global
    return countries_global


def neg_Covid_KL2(theta0):
    theta0 = param_trans2(theta0)
    global prem_global
    global cases_global
    global countries_global
    return -1 * Covid_KL2(theta0, prem_global, cases_global, countries_global)

#For testing purposes only
def pos_Covid_KL2(theta0):
    #theta0 = param_trans2(theta0)
    global prem_global
    global cases_global
    global countries_global
    return Covid_KL2(theta0, prem_global, cases_global, countries_global)

def CI_calc2(estimate, fun):
    jac = nd.Gradient(fun)
    hess = nd.Hessian(fun)
    
    return find_CI(estimate, fun, jac, hess, alpha = 0.95, 
                   disp=False, parallel = False)
    

def main():
    _test_kl_div2()

if __name__ == "__main__":
    main()
