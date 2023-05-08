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

from .utils import listify, array_shrink

def load_data(opts):
    '''
    Function that loads the data/temporal components of widefield data.
    '''
    rec_path = opts['dir'] # Path to rec
    if opts['dtype'] == 'wfield_f':
        fname = pjoin(rec_path, f'{opts["rec_name"]}.mat')
        if os.path.isfile(fname):
            f_des = io.loadmat(fname)
            data = f_des['zeromeanVc'].T # Load adjusted temporal components
        else:
            raise OSError(f'Could not find: {fname}')
    elif opts['dtype'] == 'neuron_f':
        fname = pjoin(rec_path, f'{opts["rec_name"]}.npy')
        struct = np.load(fname, allow_pickle=True)[()]
        data = struct['Y_denoised_calcium']
        # data = struct['Y_raw_fluorescence']
    elif opts['dtype'] == 'neuron_fr':
        fname = pjoin(rec_path, f'{opts["rec_name"]}.npy')
        struct = np.load(fname, allow_pickle=True)[()]
        data = struct['Y_inferred_spikes']
    else:
        raise ValueError(f'Cannot handle data of type {opts["dtype"]}')

    return data

def load_spat(opts):
    '''
    Function that loads the spatial components of widefield data.
    '''
    rec_path = opts['dir'] # Path to rec
    fname = pjoin(rec_path, f'{opts["rec_name"]}.mat')
    if os.path.isfile(fname):
        f_des = io.loadmat(fname)
        u = f_des['U'] # Load aligned spatial components
    else:
        raise OSError(f'Could not find: {fname}')
    
    return u



def load_design(opts):
    '''
    Function that loads the design DataFrame - placeholder
    '''
    rec_path = opts['dir'] # Path to rec
    if opts['dtype'] == 'wfield_f':
        fname = pjoin(rec_path, f'{opts["rec_name"]}.mat')
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
    elif opts['dtype'] == 'neuron_f' or opts['dtype'] == 'neuron_fr':
       fname = pjoin(rec_path, f'{opts["rec_name"]}.npy')
       struct = np.load(fname, allow_pickle=True)[()]
       design_df = struct['design_matrix']

    return design_df


def save_results(results, opts):
    '''
    Function to save the results.
    '''
    rec_path = opts['dir'] # Path to rec
    fname = pjoin(rec_path, f'rb_results_{opts["rec_name"]}.npz')
    np.savez(fname, **results)


def load_results(requested_objects, opts, addition=None):
    '''
    Function to load the results.
    '''
    rec_path = opts['dir'] # Path to rec
    if addition:
        fname = pjoin(rec_path, f'rb_results_{opts["rec_name"]}_{addition}.npz')
    else:
        fname = pjoin(rec_path, f'rb_results_{opts["rec_name"]}.npz')
    if not os.path.isfile(fname):
        print(f'Cannot find the file at {fname}.')
        return
    data = np.load(fname, allow_pickle=True)
    loaded_objects = []
    requested_objects = listify(requested_objects)
    for requested_object in requested_objects:
        loaded_objects.append(data[requested_object][()])
    return loaded_objects



def save_fig(fig, opts, name):
    '''
    Function to save a figure.
    '''
    rec_path = opts['dir'] # Path to rec
    fname = pjoin(rec_path, f'{name}.png')
    fig.savefig(fname)
