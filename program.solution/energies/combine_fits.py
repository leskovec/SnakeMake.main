import numpy as np
import sys
import os

import matplotlib.pyplot as plt

# read the directory for the files from the command line
if len(sys.argv) != 3:
    print('Usage: ./fit.py <energies_dir> <save_dir>')
    sys.exit(1)

energies_dir = sys.argv[1]
save_dir = sys.argv[2]

# generate a list of files in the directory energies_dir
files_all = os.listdir(energies_dir)

# split the files into 4 categories, d000, d001, d011 and d111
files={}
files['000'] = []
files['001'] = []
files['011'] = []
files['111'] = []

for file in files_all:
    if file.startswith('d000'):
        files['000'].append(file)
    elif file.startswith('d001'):
        files['001'].append(file)
    elif file.startswith('d011'):
        files['011'].append(file)
    elif file.startswith('d111'):
        files['111'].append(file)

# read the files and store the data in a numpy array
energy = {}
for key in files:
    # get number of configurations
    with open(energies_dir+'/'+files['000'][0]) as f:
        # read first line
        line = f.readline()
        # split line into words
        words = line.split()
        # number of configurations is first word
        nconf = int(words[0])
    
    energy[key] = {}
    for file in files[key]:
        # convert filename (d111_energy_tmin9-tmax22_model2exp.jack) to a key (9-22, 2exp)
        key_fit = (file.split('_')[2].replace("tmin","").replace("tmax",""),file.split('_')[3].split('.')[0].replace("model",""))
        # read the data from the file
        with open(energies_dir+'/'+file) as f:
            # read first line
            line = f.readline()
            # split line into words
            words = line.split()
            # get chi2, split on = and take second part
            chi2 = float(words[1].split('=')[1])
            nconf = int(words[0])
            # read the remaining nconf lines into a numpy array
            tmp = np.zeros((nconf))
            for konf in range(nconf):
                line = f.readline()
                words = line.split()
                tmp[konf] = float(words[1])

        # store the data in the dictionary
        energy[key][key_fit] = (tmp, chi2)

# for each key in energy loop through all key_fit in energy[key] and find the smallest chi2 amongst all key_fits
# store the corresponding fit in fit[key]
fit = {}
for key in energy:
    fit[key] = {}
    # initialize chi2_min to a large number
    chi2_min = 1000000000
    # loop through all key_fit in energy[key]
    for key_fit in energy[key]:
        # if chi2 is smaller than chi2_min, store the fit in fit[key]
        if energy[key][key_fit][1] < chi2_min:
            chi2_min = energy[key][key_fit][1]
            fit[key]['params'] = energy[key][key_fit][0]
            fit[key]['chi2'] = energy[key][key_fit][1]
            fit[key]['tmin'] = key_fit[0].split('-')[0]
            fit[key]['tmax'] = key_fit[0].split('-')[1]
            fit[key]['model'] = key_fit[1]

# print the results
for key in fit:
    print('d'+key)
    print('tmin = '+str(fit[key]['tmin']))
    print('tmax = '+str(fit[key]['tmax']))
    print('model = '+str(fit[key]['model']))
    print('chi2 = '+str(fit[key]['chi2']))
    print('params = '+str(fit[key]['params'][0]))
    print('')

# make a plot of the fit results with central values and error bars
# x axis are the key's in fit, y axis are the central value and uncertainty

# initialize lists for x and y values
x = []
y = []
y_err = []
# loop through all keys in fit
for key in fit:
    # append the key to the x list
    x.append(key)

    # append the central value to the y list
    tmp_j = fit[key]['params']

    # calculate the jackknife mean
    tmp_j_mean = np.mean(tmp_j)

    # append the jackknife mean to the y list
    y.append(tmp_j_mean)

    # calculate the jackknife error
    tmp_j_err = np.sqrt((fit[key]['params'].shape[0]-1)*np.sum((fit[key]['params']-np.mean(fit[key]['params']))**2))

    # append the jackknife error to the y_err list
    y_err.append(tmp_j_err)

# convert the lists to numpy arrays
x = np.array(x)
y = np.array(y)
y_err = np.array(y_err)

# make a plot
fig = plt.figure()
plt.errorbar(x,y,yerr=y_err,fmt='o')
plt.xlabel('d')
plt.ylabel('E')
plt.savefig('energies_chosen.pdf',bbox_inches='tight')


# upsample the jaccknife samples to non-resampled
def jack_rescale(xJ):
    n=xJ.shape[0]
    xi=np.zeros(n,dtype=np.double)
    xJavg = np.average(xJ)
    for i in range(n):
        xi[i] = xJavg - (n-1.0)*(xJ[i]-xJavg)        
    return xi

fit_nojack = {}
for key in fit:
    fit_nojack[key] = {}
    fit_nojack[key]['params'] = jack_rescale(fit[key]['params'])
    fit_nojack[key]['chi2'] = fit[key]['chi2']
    fit_nojack[key]['tmin'] = fit[key]['tmin']
    fit_nojack[key]['tmax'] = fit[key]['tmax']
    fit_nojack[key]['model'] = fit[key]['model']


# save to a .jack file
for key in fit_nojack:
    out_file = open(save_dir+'/d'+key+'_energy.jack','w')
    out_file.write(str(fit_nojack[key]['params'].shape[0])+' 1.1\n')
    for konf in range(fit_nojack[key]['params'].shape[0]):
        out_file.write('0 '+str(fit_nojack[key]['params'][konf])+'\n')
    out_file.close()




# # output the energies in .jack files
# results_jack = config.get('output', 'results_jack')
# for key in fit.keys():
#     out_file = open(results_jack+'d'+key+'_energy_'+'tmin'+str(fit_tmin)+'-tmax'+str(fit_tmax)+'_model'+FIT_MODEL+'.jack','w')
#     out_file.write(str(corr[key].shape[0])+' 1.1\n')
#     for konf in range(corr[key].shape[0]):
#         out_file.write('0 '+str(fit[key]['params'][konf][1])+'\n')
#     out_file.close()
