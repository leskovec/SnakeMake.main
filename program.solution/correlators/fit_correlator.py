# python program that reads the input file from command line. 
# input file contains information:
#      1. data file location
#      2. fit model
#      3. fit range minimum, maximum
#      4. location of output file
# the python program then does the following:
# 1. reads the data file
# 2. fits the data using the fit model
# 3. outputs the fit parameters and errors to the output file location (hdf5 format)
# 4. outputs the fit plot to the output file location (png format)
# 5. outputs the fit plot to the output file location (pdf format) 

#for the reading it uses h5py as the input file is in hdf5

#for fitting uses iminuit to fit one of the two models we can choose from:
# 1exp or 2exp, where
# 1exp is: A*exp(-E*t)
# 2exp is A*exp(-E*t) + B*exp(-E'*t)

# for plotting it uses matplotlib

# for the output file it uses h5py as the output file is in hdf5

import numpy as np
import iminuit
import matplotlib.pyplot as plt
import h5py
import sys
import argparse
import configparser


# read the input file from command line
parser = argparse.ArgumentParser(description='Fit correlator data')
parser.add_argument('-i', metavar='--i', type=str)
args = parser.parse_args()

# read the input file
config = configparser.ConfigParser()
config.read(args.i)

# read data file location
corr_file = config.get('input', 'data_filename')

# read fit range
fit_tmin = int(config.get('input', 'fit_tmin'))
fit_tmax = int(config.get('input', 'fit_tmax'))
t_list = np.arange(fit_tmin,fit_tmax+1,1)

# read corr file
tmp_corr = h5py.File(corr_file, 'r')
corr={}
print("fitting data:")
for key in tmp_corr.keys():
    tmp_d = np.real(tmp_corr[key]['lambda'][:,fit_tmin:fit_tmax+1])
    corr[key] = tmp_d
    print(key, tmp_d.shape)

print("in range: ", fit_tmin, " - ", fit_tmax)

# read fit model
FIT_MODEL = config.get('input', 'fit_model')

# constucrt covariance matrix
def make_invCorr(N,Nkonf,y_J):
    #calculate average of y_J
    y_avg = np.zeros(N,dtype=np.float64)
    for i in range(N):
        for konf in range(Nkonf):
            y_avg[i] += y_J[konf,i]
        y_avg[i] /= Nkonf

    #construct covariance matrix
    yCorr = np.zeros((N,N),dtype=np.float64)
    for i1 in range(N):
        for i2 in range(N):
            for konf in range(Nkonf):
                yCorr[i1,i2] += (y_J[konf,i1] - y_avg[i1])*(y_J[konf,i2] - y_avg[i2])

    #calculate pseudoinverse of yCorr
    invCorr=np.linalg.pinv(yCorr,rcond=1e-8)

    return invCorr

cov={}
for key in corr.keys():
    cov[key] = make_invCorr(corr[key].shape[1],corr[key].shape[0],corr[key])


# define fit models
def one_exp(t,p):
    return p['A']*np.exp(-p['E']*t)

def two_exp(t,p):
    return p['A1']*np.exp(-p['E1']*t) + p['A2']*np.exp(-p['E2']*t)

def fit_func(t,p):
    if FIT_MODEL == '1exp':
        p={'A':p[0],'E':p[1]}
        return one_exp(t,p)
    elif FIT_MODEL == '2exp':
        p={'A1':p[0],'E1':p[1],'A2':p[2],'E2':p[3]}
        return two_exp(t,p)
    else:
        print('Error: fit model not defined')
        sys.exit()

# define a correlated chi2 function
def chi2(data,model,p,t,inv_cov):
    N = inv_cov.shape[0]
    chi2_sum=0.0
    for i1 in range(N):
        for i2 in range(N):
            chi2_sum += (data[i1] - model(t[i1],p))*inv_cov[i1,i2]*(data[i2] - model(t[i2],p))


    return chi2_sum

# define a function that does the fit
def do_fit(data,t,inv_cov):
    # define the minization function
    dof = 0
    if FIT_MODEL == '1exp':
        dof = len(data) - 2
        def minF(A,E):
            p=[A,E]
            return chi2(data,fit_func,p,t,inv_cov)

        # define the minimizer
        m = iminuit.Minuit(minF, A=0.003,E=0.18)

    elif FIT_MODEL == '2exp':
        dof = len(data) - 4
        def minF(A1,E1,A2,E2):
            p=[A1,E1,A2,E2]
            return chi2(data,fit_func,p,t,inv_cov)

        m = iminuit.Minuit(minF, A1=0.003,E1=0.18,A2=0.1,E2=0.3)

    # do the minimization
    m.migrad()

    chi2_=m.fval/dof

    # return the fit parameters and errors
    return m.values,m.errors, chi2_

# do the fit
fit={}
for key in corr.keys():
    fit[key] = {}
    chi2_ = np.zeros(corr[key].shape[0],dtype=np.float64)
    if FIT_MODEL == '1exp':
        tmp_val = np.zeros((corr[key].shape[0],2),dtype=np.float64)
    elif FIT_MODEL == '2exp':
        tmp_val = np.zeros((corr[key].shape[0],4),dtype=np.float64) 
    for konf in range(corr[key].shape[0]):
        tmp_val[konf],tmp_err,chi2_[konf] = do_fit(corr[key][konf],t_list,cov[key])

    chi2_val= np.min(chi2_)
    fit[key]['chi2'] = chi2_val
    fit[key]['params'] = tmp_val

# output the energies in .jack files
results_jack = config.get('output', 'results_jack')
for key in fit.keys():
    out_file = open(results_jack+'d'+key+'_energy_'+'tmin'+str(fit_tmin)+'-tmax'+str(fit_tmax)+'_model'+FIT_MODEL+'.jack','w')
    out_file.write(str(corr[key].shape[0])+' chi2='+str(fit[key]['chi2'])+'\n')
    for konf in range(corr[key].shape[0]):
        out_file.write('0 '+str(fit[key]['params'][konf][1])+'\n')
    out_file.close()

