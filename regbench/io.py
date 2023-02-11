'''
Module containing all input-output functions.
'''

import os
from os.path import join as pjoin
import re
import itertools
import json
import numpy as np

from scipy import io
import h5py
import pandas as pd

from .utils import SVDStack

def load_data(pars):
    '''
    Function that loads the data.
    '''
    rec_path = pjoin(pars['local_disk'], pars['animal'],  'SpatialDisc', pars['rec']) # Path to recording
    if pars['dtype'] == 'widefield':
        if pars['dformat'] == 'Python':
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
                    u = np.load(fname) # If no aligned spatial components, load regular spatial components
                else:
                    raise OSError(f'Could not find: {fname}')
        elif pars['dformat'] == 'MATLAB':
            # Placeholder - load the Vc output from MATLAB
            fname = pjoin(rec_path, 'rsVc.mat')
            if os.path.isfile(fname):
                fname = pjoin(rec_path, 'demo_model.mat')
                if os.path.isfile(fname):
                    f_des = io.loadmat(fname)
                    u = f_des['U'] # Load aligned spatial components
                    svt = f_des['zeromeanVc'] # Load adjusted temporal components
            else:
                raise OSError(f'Could not find: {fname}')
        return SVDStack(u, svt)
    elif pars['dtype'] == 'Two-photon':
        pass # Placeholder for two-photon data


def load_opts(pars):
    '''
    Function that loads the options
    '''
    rec_path = pjoin(pars['local_disk'], pars['animal'],  'SpatialDisc', pars['rec']) # Path to recording
    if pars['dformat'] == 'Python':
        fname = pjoin(rec_path,'opts.json')
        if os.path.isfile(fname):
            with open(fname, 'r', encoding='utf-8') as opts_f:
                opts = json.load(opts_f) # Load options
        else:
            raise OSError(f'Could not find: {fname}')
    elif pars['dformat'] == 'MATLAB':
        fname = pjoin(rec_path, 'opts.mat')
        if os.path.isfile(fname):
            opts = io.loadmat(fname)['opts'] # Load options
            opts = {re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower(): \
                    opts[name][0][0][0].squeeze()[0] if len(opts[name][0][0][0])==0 \
                    else opts[name][0][0][0].squeeze() for name \
                    in opts.dtype.names}  # Convert to a Python dict
        else:
            raise OSError(f'Could not find: {fname}')

    return opts

def load_events(pars):
    '''
    Function that loads the events
    '''
    rec_path = pjoin(pars['local_disk'], pars['animal'],  'SpatialDisc', pars['rec']) # Path to recording
    
    if pars['dformat'] == 'Python':
        fname = pjoin(rec_path, 'events.npy')
        if os.path.isfile(fname):
            events = np.load(fname, allow_pickle=True)
        else:
            raise OSError(f'Could not find: {fname}')
    elif pars['dformat'] == 'MATLAB': # Placeholder
        fname = pjoin(rec_path, 'orgRegData.mat')
        events = {}
        if os.path.isfile(fname):
            with h5py.File(fname, 'r') as f_des: # load design matrix to isolate example events and video data 
                full_r = np.array(f_des['fullR']).T
                rec_labels = [''.join([chr(char[0]) for char in f_des[label[0]]]) for label in f_des['recLabels'] ]
                rec_labels = [re.sub(r'(?<!^)(?=[A-Z])', '_', label).lower() for label in rec_labels] # Convert to snake case
                rec_idx = np.array(f_des['recIdx'])-1 # -1 to convert to Python indexing
                idx = np.array(f_des['idx'])
                rec_idx = rec_idx[idx == 0]
                events['labels'] = ['time', 'l_vis_stim', 'r_vis_stim', 'choice', 'prev_reward', 'l_grab', 'r_grab', 'l_lick', 'r_lick', 'nose', 'whisk']
                events['types'] = [1, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3]
                events['frames'] = np.empty((np.size(full_r, 0), len(events['labels'])))
                for event_num, event_label in enumerate(events['labels']):
                    events['frames'][:, event_num] = full_r[:, np.where(rec_idx == rec_labels.index(event_label))[0][0]] # Find event frames
                events['frames'].reshape((-1, pars['frames_per_trial'], len(events['types']))) # Reshape to trials
        else:
            raise OSError(f'Could not find: {fname}')


def load_design_dframe(pars):
    '''
    Function that loads the design DataFrame - placeholder
    '''
    rec_path = pjoin(pars['local_disk'], pars['animal'],  'SpatialDisc', pars['rec']) # Path to recording
    
    fname = pjoin(rec_path, 'demo_model.mat')
    if os.path.isfile(fname):
        f_des = io.loadmat(fname)
        full_r = f_des['R'] # Load options # load design matrix to isolate example events and video data
        reg_labels = [label[0] for label in f_des['regLabels'][0] ]
        reg_labels = [re.sub(r'(?<!^)(?=[A-Z])', '_', label).lower() for label in reg_labels] # Convert to snake case
        reg_idx = np.array(f_des['regIdx'])-1 # -1 to convert to Python indexing
        design_df_columns = list(itertools.chain.from_iterable([[reg_label]*len(np.where(reg_idx == reg_id)[0]) for reg_id, reg_label in enumerate(reg_labels)]))
    else:
        raise OSError(f'Could not find: {fname}')

    design_df = pd.DataFrame(data = full_r,
                        columns = design_df_columns)

    return design_df

