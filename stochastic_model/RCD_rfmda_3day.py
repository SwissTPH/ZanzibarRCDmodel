# -*- coding: utf-8 -*-
"""
RCD_rfmda_3day.py
Changing from RCD to rfMDA (test sensitivity = 1)

Author: Aatreyee Mimi Das
"""

import numpy as np
import pandas as pd
import os
import time
import sys
import random
#%%
# First set the working directory to ZanzibarRCD
# load data
server = True #If working on a cluster vs on a local machine

if server:
    os.chdir('~/data/')
else:    
    os.chdir('~/data/')

df = pd.read_csv('parameters.csv')
print(df)

#set the fixed parameters
n = 3 #Pemba, Unguja, Mainland Tanzania (this is the order used in later arrays and matrices)
mu = df['mu']  # currently assuming 200 days for infection clearance (no treatment)
pop = df['pop']  # Total population in each patch
I_eff = np.zeros((n))
nDays = 7  # frequency of RCD implementation

# Decide whether movement from the mainland will be included or not
movement_mainland = True
if movement_mainland:
    theta = np.mat('0.990698 0.003869 0.000057; 0.003182 0.970198 0.000533; 0.00612 0.025933 0.99941')
else:
    theta = np.mat('0.990698 0.003869 0; 0.003182 0.970198 0; 0.00612 0.025933 1')

print(theta)

# Decide whether index cases or index households or neither are to be included in the prevalence dataset
index_cases = False  # everyone included including index cases and households
index_house = False  # index cases not included but index households are included
neither = True  # neither index cases nor households are included

if index_cases:
    I_eq = df['I_eq_index']
elif index_house:
    I_eq = df['I_eq_index_HH']
elif neither:
    I_eq = df['I_eq_neither']
else:
    print('Error: one of the three options needs to be selected.')
print(I_eq)

# set the RCD parameters

pcr_pos_index_HH = df['pcr_pos_index_HH'] #this does not include the index case
rdt_sensitivity = df['rdt_sensitivity']
nu = df['nu'] # number of people investigated within index HH

pcr_pos_index_HH = (pcr_pos_index_HH*(nu-1) + 1)/nu #prevalence including the index case - to ensure that index cases are treated
tau = pcr_pos_index_HH/I_eq  # targetting ratio within index households

cases_per_day = df['cases_per_day']
treatment_seeking = cases_per_day/(I_eq*pop)  
followup_days = 3 # number of days of follow up allowed can be 3, 6, 15, or 21

if followup_days==3:  # change followup_prop to proportion
    followup_prop = df['followup_3_days']
elif followup_days==6:
    followup_prop = df['followup_6_days']
elif followup_days==15:
    followup_prop = df['followup_15_days']
elif followup_days==21:
    followup_prop = df['followup_21_days']
else:
    print('Error: number of days of follow up must be 3, 6, 15 or 21.')

phi_consts = tau*nu*rdt_sensitivity
#%%
if server:
    run_number = int(sys.argv[1])-1
else:
    run_number = 1

#%%
# Form matrix M and vector p
for i in range(n):
    I_eff[i] = np.dot(pop*I_eq, np.transpose(theta[i,:]))/np.dot(pop,np.transpose(theta[i,:]))
A = np.diag(I_eff)
M = np.transpose(np.dot(A,theta))

iota = I_eq*pop*treatment_seeking*followup_prop
phi = phi_consts*iota*I_eq/pop
phi[2] = 0  # RCD not present on the mainland

p = (mu*I_eq + phi)/(1-I_eq)
print('Vector p\n',p)

# Calculate beta using the inverse of M and p

M_inv = np.linalg.inv(M)
beta = np.dot(M_inv,np.transpose(p))
print('Beta for Pemba, Unguja and Mainland Tanzania\n',beta)
beta = np.array(beta)

#%%
# Change RCD followup_prop to be 10%, 20% or the current one

df_followup = np.array((df['followup_3_days'][0],df['followup_6_days'][0],\
                        df['followup_15_days'][0], df['followup_21_days'][0]))
followup_prop_new = np.concatenate([[0., 0.1, 0.2], df_followup, [0.7, 0.8, 0.9, 1.]])

rdt_sensitivity = 1 #to simulate rfMDA
followup_prop_new = np.array([followup_prop_new,]*3).transpose()
phi_1 = np.tile(tau*nu*rdt_sensitivity, (np.shape(followup_prop_new)[0],1))
phi_consts_new = np.zeros((np.shape(followup_prop_new)[0],3))
phi_consts_new = followup_prop_new*phi_1

phi_consts_new = np.repeat(phi_consts_new, 500, axis=0)

phi_consts = phi_consts_new[run_number,:]
MaxTime = 365*40.
I0 = np.round(pop*I_eq)
S0 = pop-I0

print(I0/(I0+S0))

TS = 1 # timestep tau for tau-leap

INPUT = np.array((S0,I0))
print(INPUT)
print(INPUT[0,:], INPUT[1,:])
#%%
def stoc_eqs(INPUT, lop, countExceed): 
    Rate = np.zeros((3*n))
    Change = np.zeros((3*n,2))
    Change = Change.astype(int)
    S0 = INPUT[0,:]
    I0 = INPUT[1,:]
    N = S0+I0
    Y0 = I0/N
        
    if lop%nDays==0:
        iota = I0*treatment_seeking
        phi = phi_consts*iota
    else:
        phi = np.zeros((n))
    
    for k in range(n):
        for i in range(n):
            I_eff[i] = np.dot(N*Y0, np.transpose(theta[i,:]))/np.dot(N,np.transpose(theta[i,:]))
        
        Rate[k] = np.multiply(S0[k], np.dot(beta*I_eff, theta[:,k])); Change[k,:] = [-1, +1]  # transmission events
        Rate[n+k] = mu[k]*I0[k]; Change[n+k,:] = [+1, -1]  # recovery events
        Rate[2*n+k] = min(I0[k],nDays*phi[k]*I0[k]/N[k]); Change[2*n+k,:] = [+1, -1]  # RCD events
    Rate[3*n-1] = 0
    
    for j in range(n):
        if S0[j]==0:
            p = 0
        else:
            p = Rate[j]*TS/S0[j]
        if p>1:
            p=1
            countExceed += 1
        Draws0 = np.random.binomial(S0[j],p,1)
        INPUT[:,j] += Draws0*Change[j,:]
    for j in range(n):
        if I0[j]==0:
            p = 0
        else:
            p = Rate[n+j]*TS/I0[j]
        if p>1:
            p=1
            countExceed += 1
        Draws1 = np.random.binomial(I0[j],p,1)
        INPUT[:,j] += Draws1*Change[n+j,:]
    for j in range(n):
        if I0[j]==0:
            p = 0
        else:
            p = Rate[2*n+j]/I0[j]
        if p>1:
            p=1
            countExceed += 1
        Draws2 = np.random.binomial(I0[j],p,1)
        INPUT[:,j] += Draws2*Change[2*n+j,:]  
    return [INPUT, countExceed] 

def Stoch_Iteration(INPUT, countExceed):
    S = S0
    I = I0
    for lop in T:
        [res, countExceed] = stoc_eqs(INPUT, lop, countExceed)
        S = np.vstack((S,INPUT[0,:]))
        I = np.vstack((I,INPUT[1,:]))
        INPUT = res
    S = S[1:,:]
    I = I[1:,:]
    return [S,I,countExceed]
#%%

start = time.perf_counter()
random.seed(a=run_number)
T=np.arange(0, MaxTime, TS)
countExceed = 0
[S,I,countExceed]=Stoch_Iteration(INPUT, countExceed)

S = np.vstack((np.full(n, run_number), S))
I = np.vstack((np.full(n, run_number), I))

#save outputs
with open('../outputs/outI_RCD_rfmda_3day.csv', 'ab') as f:
    np.savetxt(f, I, fmt='%5s', delimiter=',')
f.close()


end = time.perf_counter()
print('Time taken (seconds): ', end-start)

