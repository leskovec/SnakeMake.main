from iminuit import Minuit

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import multivariate_normal


def ndec(x, offset=2):
  ans = offset - np.log10(x)
  ans = int(ans)
  if ans > 0 and x * 10. ** ans >= [0.5, 9.5, 99.5][offset]:
    ans -= 1
  return 0 if ans < 0 else ans

def StrErrNoSys_fmt(v,dv):
  # special cases
  if np.isnan(v) or np.isnan(dv):
    return '%g +- %g'%(v, dv)
  elif dv == float('inf'):
    return '%g +- inf' % v
  elif v == 0 and (dv >= 1e5 or dv < 1e-5):
    if dv == 0:
      return '0(0)'
    else:
      ans = ("%.1e" % dv).split('e')
      return "0.0(" + ans[0] + ")e" + ans[1]
  elif v == 0:
    if dv >= 9.95:
      return '0(%.0f)' % (dv)
    elif dv >= 0.995:
      return '0.0(%.1f)' % (dv)
    else:
      ndecimal = ndec(dv)
      return '%.*f(%.0f)' % (ndecimal, v, dv * 10. ** ndecimal)
  elif dv == 0:
    ans = ('%g' % v).split('e')
    if len(ans) == 2:
      return ans[0] + "(0)e" + ans[1]
    else:
      return ans[0] + "(0)"
  elif dv < 1e-6 * abs(v):
    return '%g +- %.2g' % (v, dv)
  elif dv > 1e4 * abs(v):
    return '%.1g +- %.2g' % (v, dv)
  elif abs(v) >= 1e6 or abs(v) < 1e-7:
    exponent = np.floor(np.log10(abs(v)))
    fac = 10.**exponent
    mantissa = str(v/fac)
    exponent = "e" + ("%.0e" % fac).split("e")[-1]
    return mantissa + exponent
  # normal cases
  if dv >= 9.95:
    if abs(v) >= 9.5:
      return '%.0f(%.0f)' % (v, dv)
    else:
      ndecimal = ndec(abs(v), offset=1)
      return '%.*f(%.*f)' % (ndecimal, v, ndecimal, dv)
  if dv >= 0.995:
    if abs(v) >= 0.95:
      return '%.1f(%.1f)' % (v, dv)
    else:
      ndecimal = ndec(abs(v), offset=1)
      return '%.*f(%.*f)' % (ndecimal, v, ndecimal, dv,)
  else:
    ndecimal = max(ndec(abs(v), offset=1), ndec(dv) )
    return '%.*f(%.0f)' % (ndecimal, v, dv * 10. ** ndecimal)

def rgb_to_hex(rgb):
  return '#%02x%02x%02x' % rgb

pretty_red=rgb_to_hex((192,39,45))
pretty_orange=rgb_to_hex((249,102,0))
pretty_blue=rgb_to_hex((47,122,121))
pretty_green=rgb_to_hex((65,125,10))
pretty_black=rgb_to_hex((0,0,0))
pretty_purple=rgb_to_hex((128,0,128))

prettyColors = [pretty_red,pretty_orange,pretty_blue,pretty_green,pretty_black,pretty_purple]

def set_prettyColors():
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=prettyColors) 

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
    mpl.rc('font', **font)
    mpl.rcParams.update({
                          "text.usetex": True,
                          "font.family": "serif"
                        })    
    mpl.rcParams.update({'errorbar.capsize': 2})                        

def hide_spines():
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

#------------------------------------------------------------------------------------#

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
    return i + 1

#------------------------------------------------------------------------------------#

def disp_p2(dsq, mass, xi, L):
    return ( mass**2 + dsq * ( ( 2.0*np.pi/((xi)*L) ) ** 2 ) ) ** 0.5

def disp_p2_alt(dsq, p):
    mass=p[0]
    xi=p[1]
    L=16
    psq=dsq * ( ( 2.0*np.pi/((xi)*L) ) ** 2 )
    return psq,( mass**2 + psq ) ** 0.5

#------------------------------------------------------------------------------------#

def fcn_disp_p2(dsq_in, L_in, en_v_in, en_e_in, inv_covariance, mass, xi):
  num_levels = len(dsq_in)
  diff = []
  for i in range(0,num_levels):
    diff.append(disp_p2(dsq_in[i], mass, xi, L_in[i]) - en_v_in[i])
  val = np.linalg.multi_dot([diff, inv_covariance, diff])
  return val

#------------------------------------------------------------------------------------#

def read_jack(filename):
  jack_data = []
  with open(filename, "r") as ins:
    i=0
    for line in ins:
      if (i!=0):
        cfg_val = ( (line.rstrip('\n')).split() )[1]
        jack_data.append(cfg_val)
      i+=1

  return jack_data

#------------------------------------------------------------------------------------#

def jack_mean(jack_data_):
  jack_sum = 0.0
  n_cfg = len(jack_data_)
  for i in range(0, n_cfg):
    jack_sum += float(jack_data_[i])

  return jack_sum/n_cfg

#------------------------------------------------------------------------------------#

def jack_error(jack_data_):

  n_cfg = len(jack_data_)
  avg = jack_mean(jack_data_)
  jack_resum = 0.0

  for i in range(0, n_cfg ):
    jack_cfg = []

    for j in range(0, n_cfg ):
      if(i!=j):
        jack_cfg.append(jack_data_[j])

    jack_resum += ( avg - jack_mean(jack_cfg) ) ** 2

  return ( jack_resum * (n_cfg - 1.0)/(n_cfg ) )** 0.5

#------------------------------------------------------------------------------------#

def jack_covariance(j1,j2):
    mean_j1 = jack_mean(j1)
    mean_j2 = jack_mean(j2)

    cov = 0.0

    n_cfg = min(len(j1),len(j2))
    if (n_cfg != len(j2)):
        print("error - different ensemble sizes: ", n_cfg, ", ", len(j2))
        #exit(1)

    for i in range(0, n_cfg ):
        cov += (float(j1[i])-mean_j1)*(float(j2[i])-mean_j2)

    return cov/(n_cfg*(n_cfg-1.0))

#------------------------------------------------------------------------------------#

def jack_correlation(j1,j2):
    return jack_covariance(j1,j2)/(jack_error(j1)*jack_error(j2))

#------------------------------------------------------------------------------------#

def pretty_print(mat):
  rows = mat.shape[0]
  cols = mat.shape[1]

  #print("r=",rows," c=",cols)

  print("[")
  for r in range(0,rows):

    row_out_line = "[ "

    for c in range(0,cols):
      if(c>=r):
        if (mat[r,c] > 1.e-5):
          row_out_line += ("{:7.3f}".format(mat[r,c]))
        else:
          row_out_line += ("       ")
      else:
        row_out_line += ("       ")
    #r
    print(row_out_line + " ]")
  #c
#------------------------------------------------------------------------------------#


def get_bands(func,pars,cov_pars,xmin,xmax,num):
    mn = multivariate_normal(mean=pars, cov=cov_pars)
    mn_pars=mn.rvs(size=1000, random_state=7)

    xx=np.linspace(xmin,xmax,num=num)
    pp=[]
    l_cv_f=[]
    l_sig_f=[]
    for iq in range(len(xx)):
        tmp_dat=[]
        for isam in range(len(mn_pars)):
            tmp_psq,val=func(xx[iq],mn_pars[isam])
            tmp_dat.append(val)
        l_cv_f.append(np.average(np.array(tmp_dat)))
        l_sig_f.append(np.std(np.array(tmp_dat)))
        pp.append(tmp_psq)
    l_cv_f=np.array(l_cv_f)
    l_sig_f=np.array(l_sig_f)
    f_min = l_cv_f-l_sig_f
    f_plu = l_cv_f+l_sig_f
    return f_min,f_plu,pp

def read_spectrum_list(spectrum_list_file, path_prefix):
  n_lines = file_len(spectrum_list_file)
  print("reading ", spectrum_list_file, ", found n =", n_lines," lines")

  with open(spectrum_list_file, "r") as ins:

    dsq_vals = []
    L_vals = []
    en_vals = []
    en_errs = []
    jack_files = []
    jack_data = []

    for spec_line in ins:
        #print(spec_line.rstrip('\n'))
        this_line = (spec_line.rstrip('\n')).split()

        if (this_line[0]=="#"):
            #commented out lines begin with "# "
            continue

        if ( len(this_line) != 3 ):
            print("bad data - not enough elements on line")

        L     = int(this_line[0])
        psq  =  int(this_line[1])
        jackf = path_prefix + this_line[2]

        jack_tmp = read_jack(jackf)

        dsq_vals.append(psq)
        L_vals.append(L)
        jack_files.append( jackf )
        jack_data.append( jack_tmp )
        en_vals.append( jack_mean( jack_tmp ) )
        en_errs.append( jack_error( jack_tmp ) )

    n_data = len(jack_files)

    correlation = np.zeros( (n_data,n_data) )
    covariance  = np.zeros( (n_data,n_data) )

    for x in range(0,len(dsq_vals)):
        for y in range(x,len(dsq_vals)):
            if (L_vals[x]!=L_vals[y]):
                covariance[x,y]=0.0
                covariance[y,x]=0.0
                correlation[x,y]=0.0
                correlation[y,x]=0.0
            else:
                covariance[x,y] = jack_covariance(jack_data[x],jack_data[y])
                covariance[y,x] = covariance[x,y]

                correlation[x,y] = covariance[x,y]/(en_errs[x]*en_errs[y])
                correlation[y,x] = correlation[x,y]

    print("correlation matrix is:")
    pretty_print(correlation)

  return [dsq_vals, L_vals, en_vals, en_errs, covariance, correlation]

def dispersion_plot(mass, xi, mass_syst_err, xi_syst_err, dsq_vals, L_vals, en_vals, en_errs,covMat, name, chisq, limits):
    xmin=limits[0]
    xmax=limits[1]
    ymin=limits[2]
    ymax=limits[3]

    max_psq = 0.0
    psq_vals=[]
    for i in range(0,len(dsq_vals)):
        this_psq = dsq_vals[i]*(2.0*np.pi/(xi*L_vals[i]))**2

        if (this_psq > max_psq):
            max_psq = this_psq

        psq_vals.append( this_psq )
    num=500
    psq = np.linspace(0, max_psq*1.2, num) # for fit lines

    dsq=np.linspace(np.min(dsq_vals),np.max(dsq_vals),num)

    plt.figure(figsize=(6, 11))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.subplot(211)

    plotLabel = r"$m$ = " + ("{:7.5f}".format(mass)) + r"$\pm$" + ("{:7.5f}".format(mass_syst_err)) + r"$\quad\xi$ = " + ("{:6.4f}".format(xi)) + r"$\pm$" + ("{:6.4f}".format(xi_syst_err))
    #psq/((2.0*pi/(xi*16))**2
    disp_minus,disp_plus,pp =get_bands(disp_p2_alt,[mass,xi],covMat,min(dsq),max(dsq),num)
    plt.fill_between(pp,disp_minus,disp_plus,color=pretty_red,alpha=0.5)

    plt.xlabel(r'$p^2$')
    plt.ylabel(r'$E$')

    L_L = sorted(set(L_vals)) #index of L's, eg 16, 20, 24
    num_Ls = len(L_L) #sort -u L_vals | wc -l
    #L_L = [16, 20, 24]
    L_symb = ['o','s','p','^','v','h']

    for l in range(0,num_Ls):
        psq_vals_L =[]
        en_vals_L = []
        en_errs_L = []
        for i in range(0,len(dsq_vals)):
            if (L_vals[i] == L_L[l]):
                psq_vals_L.append(psq_vals[i])
                en_vals_L.append(en_vals[i])
                en_errs_L.append(en_errs[i])
        #-- i --#

        plt.errorbar(psq_vals_L, en_vals_L, yerr=en_errs_L, markersize=4,fmt=L_symb[l], mfc='white',color='black',mec='black',elinewidth=2.5, capsize=4.5, mew=1.3)
    # -- l -- #
 
    hide_spines()

    # add info text
    infoTextZero = r"$ E = \sqrt{ m^2 + (p/\xi)^2}$"
    infoTextOne = r"$ m = "+StrErrNoSys_fmt(mass,mass_syst_err)+"$"
    infoTextTwo = r"$ \xi = "+StrErrNoSys_fmt(xi,xi_syst_err)+"$"
    infoTextThree = r"$\frac{\chi^2}{{\rm dof}} ="+"{0:3.2f}".format(chisq)+"$"
    plt.text(1.02*np.abs(xmin),ymax,infoTextZero)
    plt.text(1.02*np.abs(xmin),0.91*ymax,infoTextOne)
    plt.text(1.02*np.abs(xmin),0.82*ymax,infoTextTwo)
    plt.text(0.75*xmax,1.1*ymin,infoTextThree)
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)

    plt.subplot(212)

    plt.plot([xmin,xmax], [0,0]) # zero line

    plt.xlabel(r'$p^2$')
    plt.ylabel(r'$E_{lat}-E_{fit}$')

    max_ed = 0.0
    for l in range(0,num_Ls):
        psq_vals_L =[]
        ed_vals_L = []
        en_errs_L = []
        for i in range(0,len(dsq_vals)):
            if (L_vals[i] == L_L[l]):
                this_ed = en_vals[i] - disp_p2(psq_vals[i]/((2.0*np.pi/(xi*L_vals[i]))**2), mass, xi, L_vals[i])
                if (abs(this_ed) >  max_ed):
                    max_ed = abs(this_ed)

                psq_vals_L.append(psq_vals[i])
                ed_vals_L.append( this_ed )
                en_errs_L.append(en_errs[i])
        #-- i --#
        plotLab = str(L_L[l]) + r"$^3$"
        plt.errorbar(psq_vals_L, ed_vals_L, yerr=en_errs_L, markersize=3,fmt=L_symb[l],color='k', mfc='white',mec='k',elinewidth=1, capsize=2, mew=0.7, label=plotLab)
    #-- l --#

    plt.ylim([-max_ed*1.2,max_ed*1.2])
    plt.xlim(xmin,xmax)

    hide_spines()


    plt.savefig("dispersion_"+name+".pdf", bbox_inches='tight')



def make_dispersion(input_spectrum_list,path_prefix,name,limits):
    dsq_vals = []
    L_vals = []
    en_vals = []
    en_errs = []
    jack_files = []
    correlation = []
    covariance  = []

    [dsq_vals, L_vals, en_vals, en_errs, covariance, correlation] = read_spectrum_list  (input_spectrum_list, path_prefix)

    print("dispersion fit with ", len(en_vals), " levels")

    inv_covariance = np.linalg.inv(covariance)

    def minFunc(mass, xi):
        return fcn_disp_p2(dsq_vals, L_vals, en_vals, en_errs, inv_covariance, mass, xi)
    
    m = Minuit(minFunc, mass=0.3, xi=1)
    m.migrad()
    chisq = m.fval/(float(len(dsq_vals))-2.0)

    for i in range(len(m.values)):
        print("values: ",StrErrNoSys_fmt(m.values[i],m.errors[i]))
    print ("chisq/N_dof =", "{:8.5f}".format(m.fval), "/(", len(dsq_vals) ,"- 2) =", "{:8.5f}   ".format(chisq), "\n")
    print("cov")
    print(m.covariance)

    sigmas = []
    print("L  dsq psq     En_Lat   Er_Lat   En_disp  Sigma")
    for i in range(0,len(dsq_vals)):
        result = disp_p2(dsq_vals[i], m.values["mass"], m.values["xi"], L_vals[i])
        psq = dsq_vals[i]*(2.0*np.pi/(m.values["xi"]*L_vals[i]))**2
        sigmas.append( (float((en_vals[i]-result)/en_errs[i])) )
        print(L_vals[i], dsq_vals[i], "{:8.4f}".format(psq), "{:8.5f}".format(float(en_vals[i])),   "{:8.5f}".format(float(en_errs[i])), "{:8.5f}".format(float(result)), "{:6.2f}".format( (float    ((en_vals[i]-result)/en_errs[i])) ))
    # -- end i -- #

    par_cov=m.covariance
    dispersion_plot(m.values["mass"], m.values["xi"], m.errors["mass"], m.errors["xi"], dsq_vals, L_vals, en_vals, en_errs,par_cov,name,chisq,limits)