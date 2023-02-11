# MIT License

# Copyright (c) 2019 Churchland laboratory

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings

import numpy as np
from scipy import linalg as LA
import pandas as pd
from tqdm.notebook import tqdm

def make_design_dframe(events, opts):
    '''
    This function generates a design matrix from a column matrix with binaryevents.
    event_types defines the type of design matrix that is generated.
    (1 = full trial, 2 = post-event, 3 = peri-event)
    Originally written in MATLAB by Simon Musall, 2019
    Adapted to Python and modified by Michael Sokoletsky, 2023
    '''
    frames = opts['frames_per_trial']
    event_frames = events['frames'].reshape((-1, frames, len(events['types']))) # Reshape to trials
    trial_cnt = np.size(event_frames, 0) # Num of trials
    design_df = [None] *  len(events['types'])
    for i_reg, (event_type, event_label) in tqdm(enumerate(zip(events['types'], events['labels'])),
                                             total=len(events['types']),
                                             desc = 'Building design matrix'):
        # Run over trials
        d_mat = [None] * trial_cnt
        for i_trial in range(trial_cnt):
            # Determine index for current event type and trial
            if event_type == 1:
                kernel_idx = np.arange(frames) # Index up to the number of frames
            elif event_type == 2:
                kernel_idx = np.arange(np.ceil(opts['s_post_time'] * opts['fs']).astype(int)) # Index for design matrix to cover post event activity
            elif event_type == 3:
                kernel_idx = np.arange(-np.ceil(opts['m_pre_time']* opts['fs']).astype(int),np.ceil(opts['m_post_time']* opts['fs']).astype(int))
            else:
                print('Unknown event type. Must be a value between 1 and 3.')
            # Fetch the zero lag regressor
            trace = event_frames[i_trial, :, i_reg].astype(bool)
            # Create full design matrix
            c_idx = np.where(trace)+kernel_idx[:,np.newaxis]
            c_idx = np.clip(c_idx,-1,frames-1)
            c_idx = c_idx + np.arange(0,frames*len(kernel_idx), frames)[:, np.newaxis]
            c_idx[c_idx < 0] = frames-1
            c_idx[c_idx > (frames*len(kernel_idx) - 1)] = frames*len(kernel_idx) - 1
            d_mat[i_trial] = np.zeros((frames,len(kernel_idx)))
            d_mat[i_trial][c_idx % frames,c_idx // frames] = True
            d_mat[i_trial][-1,:] = False #  Don't use last timepoint of design matrix to avoid confusion with indexing.
            d_mat[i_trial][-1,1:] = d_mat[i_trial][-2,:-1] #  Replace with shifted version of previous timepoint

        design_df[i_reg] = pd.DataFrame(data = np.vstack(d_mat),
                             columns = [event_label] *len(kernel_idx) )
        zero_reg = (design_df[i_reg] != 0).any(axis=0)
        design_df[i_reg] = design_df[i_reg].loc[:, zero_reg] # Remove empty regressors

    design_df = pd.concat(design_df, axis=1) # Combine all regressors into larger matrix

    return design_df


def calc_regressor_orthogonality(R, idx, rmv = True):
    '''
    This function calculates regressor orthogonality.
    '''
    qrr = LA.qr(np.divide(R,np.sqrt(np.sum(R**2,0))),mode='r') # orthogonalize normalized design matrix
    
    if np.sum(abs(np.diagonal(qrr)) > np.max(np.shape(R)) * abs(np.spacing(qrr[0,0]))) < np.size(R,1): # check if design matrix is full rank
        if rmv:
            keep_idx = abs(np.diagonal(qrr)) > max(np.shape(R)) * abs(np.spacing(qrr[0,0])) # reject regressors that cause rank-defficint matrix
            warnings.warn(f'Warning: design matrix contains redundant regressors! Removing {np.sum(~keep_idx)}/{np.size(R,1)} regressors.')
            R = R[:,keep_idx]
            idx = idx[keep_idx]
        else:
            warnings.warn('Warning: design matrix contains redundant regressors! This will break the model.')           
                      
    return qrr, R, idx