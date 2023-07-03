from iminuit import Minuit

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import multivariate_normal

from dispersion import *

set_prettyColors()

input_spectrum_list = "file.list" 
path_prefix = ""
limits=[-0.002,0.15,0.17,0.45]
make_dispersion(input_spectrum_list,path_prefix,"pion",limits)
