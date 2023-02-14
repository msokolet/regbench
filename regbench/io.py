'''
Module containing all input-output functions.
'''

import os
from os.path import join as pjoin
import re
import itertools
import numpy as np

from scipy import io
import pandas as pd

from .utils import SVDStack

def load_data(opts):
    '''
    Function that loads the data.
    '''
    rec_path = pjoin(opts['local_disk'], opts['animal'],  'SpatialDisc', opts['rec']) # Path to rec
    if opts['dformat'] == 'Python':
        fname = pjoin(rec_path,'SVTcorr.npy')
        if os.path.isfile(fname):
            svt = np.load(fname) # Load adjusted temporal components
        else:
            raise OSError(f'Could not find: {fname}')
        fname = pjoin(rec_path,'U_atlas.npy')
        if os.path.isfile(fname):
            u = np.load(fname) # Load aligned spatial components
        else:
            fname = pjoin(rec_path,'U.npy')
            if os.path.isfile(fname):
                u = np.load(fname) # If no aligned components, load regular spatial components
            else:
                raise OSError(f'Could not find: {fname}')
    elif opts['dformat'] == 'MATLAB':
        # Placeholder - load the Vc output from MATLAB
        fname = pjoin(rec_path, 'rsVc.mat')
        if os.path.isfile(fname):
            fname = pjoin(rec_path, 'demo_model.mat')
            if os.path.isfile(fname):
                f_des = io.loadmat(fname)
                u = f_des['U'] # Load aligned spatial components
                svt = f_des['zeromeanVc'].T # Load adjusted temporal components
        else:
            raise OSError(f'Could not find: {fname}')
        return SVDStack(u, svt)


def load_design(opts):
    '''
    Function that loads the design DataFrame - placeholder
    '''
    rec_path = pjoin(opts['local_disk'], opts['animal'],  'SpatialDisc', opts['rec']) # Path to rec
    fname = pjoin(rec_path, 'demo_model.mat')
    if os.path.isfile(fname):
        f_des = io.loadmat(fname)
        full_r = f_des['R'] # Load design matrix
        reg_labels = [label[0] for label in f_des['regLabels'][0] ]
        reg_labels = [re.sub(r'(?<!^)(?=[A-Z])', '_', label).lower() \
                     for label in reg_labels] # Convert to snake case
        reg_idx = np.array(f_des['regIdx'])-1 # -1 to convert to Python indexing
        design_df_columns = list(itertools.chain.from_iterable([[reg_label]*\
                                 len(np.where(reg_idx == reg_id)[0]) \
                                 for reg_id, reg_label in enumerate(reg_labels)]))
    else:
        raise OSError(f'Could not find: {fname}')

    design_df = pd.DataFrame(data = full_r,
                        columns = design_df_columns)

    return design_df

def save_results(results, opts):
    '''
    Function to save the results.
    '''
    rec_path = pjoin(opts['local_disk'], opts['animal'],  'SpatialDisc', opts['rec']) # Path to rec
    fname = pjoin(rec_path, 'results.npz')
    np.savez(fname, **results)

def save_fig(fig, opts, name):
    '''
    Function to save a figure.
    '''
    rec_path = pjoin(opts['local_disk'], opts['animal'],  'SpatialDisc', opts['rec']) # Path to rec
    fname = pjoin(rec_path, f'{name}.svg')
    fig.savefig(fname)
