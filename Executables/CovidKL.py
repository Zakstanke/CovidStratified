# -*- coding: utf-8 -*-
"""
COVID Kullback-Leibler
Created on Wed Jan 12 19:17:00 2022

@author: zakst
"""

import numpy as np
from scipy import optimize, stats
from math import isclose
import math
import warnings
import CovidEig

#import os
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


# Returns Kullback-Leibler Divergence 
def kl_div(p, q, logbase = math.e):    
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
    
    val = 0
    for i in range(0, len(p)):
        val = val + (p[i] * math.log((p[i]/q[i]), logbase)) # defaults to natural log
    assert(val >= 0), "Calculated KL Divergence is negtive: (%d)" % val
    #val = math.log(val, 10)
    return val

# Tests known calculation for Kullback-Leibler Divergence
# Just runs through a simple test case
def _test_kl_div():
    p_test = [0.10, 0.40, 0.50]
    q_test = [0.80, 0.15, 0.05]
    # Check accuracy to 6 decimal places
    temp = kl_div(p_test, q_test, 2)
    assert (isclose(temp, 1.926979, rel_tol=1e-6)), temp
    temp = kl_div(q_test, p_test, 2)
    assert (isclose(temp, 2.021647, rel_tol=1e-6)), temp
    return "Test executed successfully"

# Per-Country calculation from equation 7 from pickling
# total cases per country times kl divergence: (n_tot * D(u_hat || u_tilda))
# theta0 is a 14 length array, containing two 7-length arrays for s & v
def Covid_KL_k(theta0, prem_in, kcases_in, split = 2, debug = False):
    
    # Input validation
    # Return infinity if there are bad values to punish minimizer
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
    kcases = CovidEig.case_dim_split(kcases_in, ignore_last=True)
    for i in range(0,len(kcases)):
        total_cases = total_cases + kcases[i] #Find total cases per country
    u_hat = [element / total_cases for element in kcases]
    
    u_tilda = CovidEig.scv_eig(s = theta0[0:(int(len(theta0)/2))], c = prem_in, v = theta0[(int(len(theta0)/2)):],
                               split = split, debug = debug)
    if (u_tilda == 0): return math.inf #Check for fail condition within scv_eig
    assert(isclose(sum(u_tilda), 1, rel_tol=1e-3)), "u-tilda: %s,\tsum: %s" %(u_tilda, sum(u_tilda))
    assert(isclose(sum(u_hat), 1, rel_tol=1e-3)), "u-hat: %s,\tsum: %s" %(u_hat, sum(u_hat))
    
    return kl_div(u_hat, u_tilda)


# Equation 7 from pickling
# Sums per-country calculations from previous function
# cases_in -> pandas dataframe, prem_in -> dictionary, country_codes -> string array
def Covid_KL(theta_in, prem_in, cases_in, country_codes, debug = False):
    kl_sum = 0
    #Sum the countries together
    for i in range(0, len(country_codes)):
        # Asserts check that the letter codes can be used on both the cases dataframe and prem dictionary
        # Unlikely the asserts themselves will be triggering, but the call inside the function will throw its own errors if something is wrong
        #assert(prem_in[country_codes[i]].any != None)
        #assert(cases_in[country_codes[i]].any != None)
        res = Covid_KL_k(theta_in, prem_in[country_codes[i]], cases_in[country_codes[i]], debug = debug)
        kl_sum = kl_sum + res
        
    return kl_sum


def main():
    _test_kl_div()

if __name__ == "__main__":
    main()
