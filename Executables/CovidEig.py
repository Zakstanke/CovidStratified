# -*- coding: utf-8 -*-
"""
Covid Eig & matrix functions
Created on Wed Jul 20 15:09:58 2022

@author: zakst
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:35:06 2022

@author: zakst
"""

import numpy as np
from numpy import linalg as la
from math import isclose
import math
import warnings


def case_dim_split(list_in, split = 2, ignore_last = True):
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


# Recieves array of 7 (all age groups minus one), outputs diagonal 16x16 matrix
def theta_builder(array_in, stand_age = 2, split = 2, diag = True):
    if (len(array_in) != 7): warnings.warn("Dimentions not equal to 7")
    # Insert value for standardized age group so that everything scales around it
    theta = np.insert(array_in, stand_age, 1)
    
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
def scv_eig(s, c, v, stand_age = 2, split = 2, debug = False, peek = False):
    # Turn s & v into square matricies    
    s_in = s
    v_in = v
    if (split > len(s)): warnings.warn("Standardizing age group out of bounds")
    s = theta_builder(s, stand_age = stand_age, split = split)
    v = theta_builder(v, stand_age = stand_age, split = split)
        
    # Ensure that all matricies are the same square dimensions
    assert (len(c) == len(c[0])), "not square"
    assert ((len(s) == len(c)) & (len(c) == len(v))), "rows not equal"
    assert ((len(s[0]) == len(c[0])) & (len(c[0]) == len(v[0]))), "cols not equal"
    
    #Create SCV Matrix
    scv = np.matmul(np.matmul(s,c),v)
    
    # Check SCV meets assumptions
    neg = False
    pos = False
    for i in range(0,len(scv)):
        for j in range(0,len(scv[0])):
            if (scv[i,j] < 0): neg = True
            elif (scv[i,j] > 0): pos = True
            #assert(scv[i,j] != 0), scv[i,j]
            elif (scv[i,j] == 0): return 0
            else: assert(False), "Unknown number detected in SCV: (%f)" % scv[i,j] 
    assert((neg == True) or (pos == True))
    assert ((neg == True) != (pos == True)), "SCV matrix contains both negative and positive elements"
    
    # Find Perron Frobenius Eigenvalue/vector
    eigs = la.eig(scv)
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
    if (neg == pos):
        if (debug):
            warnings.warn("Eigenvector contains both negative and positive elements")
            
    if (neg): eig = (-1 * eig) # If negative, flip eigvector so all elements are positive
    
    # Scale eigenvector so it sums to 1
    eig_sum = sum(eig)
    eig = [x/eig_sum for x in eig] 
    
    eig0 = eig # This var is so we can look at the eig at this point in time during debugging
    
    # Remove complex element from floats
    for i in range(0, len(eig)):
        eig[i] = eig[i].real
    
    assert(sum(np.iscomplex(eig)) == 0), eig #Assert real
    if not(sum((i > 0) for i in eig) >= len(eig)):
        if (debug):
            print("\nS:\t", s_in)
            #print("C:\t", c)
            print("V:\t", v_in)
            print("SCV:\t", scv)
            print("Eig:\t", eig0)
            #print(type(eig[1]))
    assert(sum((i > 0) for i in eig) >= len(eig)), eig #Assert positive
    assert(isclose(sum(eig), 1, rel_tol=1e-6)), sum(eig) #Assert sums to 1
    
    if (peek):
        print("\nS:\t", s_in)
        #print("C:\t", c)
        print("V:\t", v_in)
        print("SCV:\t", scv)
        print("Eig:\t", eig0)
        #print(type(eig[1]))
    
    return eig

def main():
    return("Make some test cases just for this file")

if __name__ == "__main__":
    main()