# -*- coding: utf-8 -*-
"""
COVID Estimization-Maximization
Created on Wed Jul 20 15:16:26 2022

@author: zakst
"""

import numpy as np
from math import isclose
import math
import warnings
import CovidEig


def chi_approx(uh, ut):
    assert(len(uh) == len(ut)), "vectors not the same size"
    if ((len(uh) != 8) and (len(uh) != 16)): 
        warnings.warn("Unexpected vector size")
        print(len(uh))
    
    result = 0
    for i in range(0, len(uh)):
        result = result + ((uh[i] - ut[i])**2)/ut[i]
        
    return result


def Covid_reg_k(theta0, prem_in, kcases_in, stand_age = 2, split = 2, debug = False):
    
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
                               stand_age = stand_age, split = split, debug = debug)
    if (u_tilda == 0): return math.inf #Check for fail condition within scv_eig
    assert(isclose(sum(u_tilda), 1, rel_tol=1e-3)), "u-tilda: %s,\tsum: %s" %(u_tilda, sum(u_tilda))
    assert(isclose(sum(u_hat), 1, rel_tol=1e-3)), "u-hat: %s,\tsum: %s" %(u_hat, sum(u_hat))
    
    chisq = total_cases * chi_approx(u_hat, u_tilda)
    
    return chisq


def smoothing(theta):
    smooth_s = 0
    smooth_v = 0
    s = theta[0:(int(len(theta)/2))]
    v = theta[(int(len(theta)/2)):]
    
    for i in range(1, len(s)-1):
        smooth_i = (s[i-1] - 2*s[i] + s[i+1])
        smooth_s = smooth_s + (smooth_i * smooth_i)
        
    for j in range(1, len(v)-1):
        smooth_j = (v[j-1] - 2*v[j] + v[j+1])
        smooth_v = smooth_v + (smooth_j * smooth_j)
    return smooth_s + smooth_v

def _test_smoothing():
    test_theta = [0.219294308, 0.423309799, 1.124597388, 1.511734653, 1.560386081, 1.727919503, 1.734579037, 
                  3.77747199,  0.71170154,  0.769347828, 0.104869899, 2.090248998, 1.98923028,  3.373799478]
    assert(math.isclose(smoothing(test_theta), 24.359521043490876,
                        rel_tol = 1e-6))
    return "Smoothing executed successfully"


def Covid_reg(theta_in, prem_in, cases_in, country_codes, stand_age = 2, split = 2, Lambda = (42/12), debug = False):
    smooth = smoothing(theta_in)
    result = 0
    #Sum the countries together
    for i in range(0, len(country_codes)):
        # Asserts check that the letter codes can be used on both the cases dataframe and prem dictionary
        # Unlikely the asserts themselves will be triggering, but the call inside the function will throw its own errors if something is wrong
        #assert(prem_in[country_codes[i]].any != None)
        #assert(cases_in[country_codes[i]].any != None)
        res = Covid_reg_k(theta_in, prem_in[country_codes[i]], cases_in[country_codes[i]], 
                          stand_age = stand_age, split = split, debug = debug)
        
        result = result + res
        
    return result + (Lambda * smooth)



def main():
    _test_smoothing()

if __name__ == "__main__":
    main()
    