#  wfield - tools to analyse widefield data - plotting the results / summary plots 
# Copyright (C) 2020 Joao Couto - jpcouto@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import copy
from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt

def plot_regressor_orthogonality(QRR, localdisk = None, title = 'Regressor orthogonality', figsize = [9,5]):
    '''
    Plots regressor orthogonality. The resulting plot ranges from 0 to 1 for each regressor, with 1 being
    fully orthogonal to all preceeding regressors in the matrix and 0 being
    fully redundant
    QRR             : Output of calc_regressor_orthogonality
    localdisk       : Output folder (figure will be localdisk/regressor_orthogonality.pdf)

    '''
    
    fig = plt.figure(figsize=figsize)

    plt.plot(abs(np.diagonal(QRR)),linewidth=2)
    plt.ylim(0,1.1)

    plt.title(title) # this shows how orthogonal individual regressors are to the rest of the matrix

    plt.ylabel('Norm. vector angle')
    plt.xlabel('Regressors')

    if not localdisk is None:
        folder = localdisk
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fig.savefig(pjoin(folder,'regressor_orthogonality.pdf'))
        
        
def plot_model_corr(cvR2, title, c_max = 0.5, localdisk = None):
    '''
    Plots cross-validated R^2 results
    cvR2            : Cross-validated R^2 array
    title       : Name of the regressor or regressor category used in constructing the model
    localdisk       : Output folder (figure will be localdisk/model_corr.pdf)

    '''
    curr_cmap = copy.copy(plt.cm.get_cmap('inferno'))
    curr_cmap.set_bad(color='white') # make nan values white

    fig = plt.figure(figsize=[9,5])

    plt.imshow(cvR2,cmap=curr_cmap, vmin=0,vmax=c_max);
    
    plt.title(f'cVR$^2$ - {title}') # this shows how orthogonal individual regressors are to the rest of the matrix
    plt.axis('off')

    if not localdisk is None:
        folder = localdisk
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fig.savefig(pjoin(folder,f'{title}_model_corr.pdf'))
        


