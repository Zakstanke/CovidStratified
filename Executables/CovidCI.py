# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:35:06 2022

@author: zakst
"""

import os
import numpy as np
import warnings
import numdifftools as nd
from ci_rvm import find_CI
import CovidEM

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


# Applies exponential transformation to parameters
def param_trans(params_in):
    if (type(params_in) == type(np.float64(1.11))):
        return np.exp(params_in)
    params = [0]*len(params_in)
    for i in range(0,len(params_in)):
        params[i] = np.exp(params_in[i])
    return params

def neg_Covid_KL(theta0):
    #theta0 = param_trans(theta0)
    global prem_global
    global cases_global
    global countries_global
    return -1 * CovidEM.Covid_KL(theta0, prem_global, cases_global, countries_global)

#For testing purposes only
def pos_Covid_KL(theta0):
    #theta0 = param_trans(theta0)
    global prem_global
    global cases_global
    global countries_global
    return CovidEM.Covid_KL(theta0, prem_global, cases_global, countries_global)

def CI_calc(estimate, fun):
    jac = nd.Gradient(fun)
    hess = nd.Hessian(fun)
    
    return find_CI(estimate, fun, jac, hess, alpha = 0.95, disp=True)


def main():
    
    return "something amazing"

if __name__ == "__main__":
    main()
    