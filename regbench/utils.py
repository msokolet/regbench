'''
Module containing utility classes and functions.
'''

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class SVDStack(object):
    '''
    This class created the image stack for widefield imaging data.
    '''

    def __init__(self, u, svt, dims=None, dtype='float32'):
        self.u = u.astype('float32')
        self.svt = svt.astype('float32')
        self.issparse = False
        self.mask = np.isnan(u[:, :, 0])  # create the mask
        if issparse(u):
            self.issparse = True
            if dims is None:
                raise ValueError(
                    'Supply dims = [H,W] when using sparse arrays')
            self.u_flat = self.u
        else:
            if dims is None:
                dims = u.shape[:2]
            self.u_flat = array_shrink(u, self.mask)
        self.shape = [svt.shape[1], *dims]
        self.dtype = dtype

    def __len__(self):
        return self.svt.shape[0]


def vis_score(data, m_svt, opts, frame_idx, dtype='float32'):
    '''
    Short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, u is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.

    Originally written in MATLAB by Simon Musall, 2019

    Adapted to Python by Michael Sokoletsky, 2021
    '''

    if opts['map_met'] == 'r2':
        if opts['sample_trials'] > 0:
            Vc = data.svt[frame_idx, :].T
            Vm = m_svt[frame_idx, :].T
        else:
            Vc = data.svt.T
            Vm = m_svt.T
        cov_Vc = np.cov(Vc, dtype=dtype)  # S x S
        cov_Vm = np.cov(Vm, dtype=dtype)  # % S x S
        c_cov_V = (Vm - np.expand_dims(np.mean(Vm, 1), axis=1)
                ) @ Vc.T / (np.size(Vc, 1) - 1)  # S x S
        cov_P = np.expand_dims(np.sum((data.u_flat @ c_cov_V) * data.u_flat, 1), axis=0)  # 1 x P
        var_P1 = np.expand_dims(np.sum((data.u_flat @ cov_Vc) * data.u_flat, 1), axis=0)  # 1 x Pii
        var_P2 = np.expand_dims(np.sum((data.u_flat @ cov_Vm) * data.u_flat, 1), axis=0)  # 1 x P
        std_Px_Py = var_P1 ** 0.5 * var_P2 ** 0.5  # 1 x P
        corr_mat = (cov_P / std_Px_Py).T
        corr_mat = array_shrink(corr_mat, data.mask, 'split') ** 2
    elif opts['map_met'] == 'R2':
        if opts['sample_trials'] > 0:
            real_act = data.svt[frame_idx, :] @ data.u_flat.T
            model_act = m_svt[frame_idx, :] @ data.u_flat.T
        else:
            real_act = data.svt @ data.u_flat.T
            model_act = m_svt @ data.u_flat.T
        scores = r2_score(real_act, model_act, multioutput='raw_values')
        corr_mat = array_shrink(scores, data.mask, 'split')
        
    return corr_mat


def new_vis_score(data, m_svt, opts, frame_idx):
    '''
    Short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, u is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.

    Originally written in MATLAB by Simon Musall, 2019

    Adapted to Python by Michael Sokoletsky, 2021
    '''
    if opts['sample_trials'] > 0:
        y_true = data.svt[frame_idx, :]
        y_pred = m_svt[frame_idx, :]
    else:
        y_true = data.svt
        y_pred = m_svt
    y_true_centered = y_true - np.mean(y_true, axis=0)
    y_pred_centered = y_pred - np.mean(y_pred, axis=0)
    # compute covariances
    cov_true_pred = np.einsum('ij,kj,kp,ip->i', data.u_flat, y_true_centered, y_pred_centered, data.u_flat, optimize="greedy")
    cov_true = np.einsum('ij,kj,kp,ip->i', data.u_flat, y_true_centered, y_true_centered, data.u_flat, optimize="greedy")
    cov_pred = np.einsum('ij,kj,kp,ip->i', data.u_flat, y_pred_centered, y_pred_centered, data.u_flat, optimize="greedy")
    # output r2 map
    scores = cov_true_pred ** 2 / (cov_true * cov_pred)
    corr_mat = array_shrink(scores, data.mask, 'split')

    return corr_mat

def array_shrink(data_in, mask, mode='merge'):
    '''
    Code to merge the first two dimensions of matrix 'DataIn' into one and remove
    values based on the 2D index 'mask'. The idea here is that DataIn is a stack
    of images with resolution X*Y and pixels in 'mask' should be removed to
    reduce datasize and computional load of subsequent analysis. The first 
    dimension of 'DataOut' will be the product of the X*Y of 'DataIn' minus 
    pixels in 'mask'. 
    Usage: data_out = array_shrink(data_in,mask,'merge')

    To re-assemble the stack after computations have been done, the code
    can be called with the additional argument 'mode' set to 'split'. This
    will reconstruct the original data structure removed pixels will be
    replaced by NaNs.
    Usage: data_out = array_shrink(data_in,mask,'split')

    Originally written in MATLAB by Simon Musal, 2016

    Adapted to Python by Michael Sokoletsky, 2021
    '''

    d_size = np.shape(data_in)  # size of input matrix
    if d_size[0] == 1:
        data_in = np.squeeze(data_in)  # remove singleton dimensions
        d_size = np.shape(data_in)

    if len(d_size) == 2:
        if d_size[0] == 1:
            data_in = data_in.T
            d_size = np.shape(data_in)  # size of input matrix

        d_size = d_size + (1,)

    if mode == 'merge':  # merge x and y dimension into one

        data_in = np.reshape(data_in,
                             (np.size(mask),
                              np.prod(d_size[mask.ndim:])))
        mask = mask.flatten()  # reshape mask to vector
        data_in = data_in[~mask, :]
        orig_size = [np.size(data_in, 0), *d_size[2:]]
        data_in = np.reshape(data_in, tuple(orig_size))
        data_out = data_in

    elif mode == 'split':  # split first dimension into x- and y- dimension based on mask size

        # check if datatype is single. If not will use double as a default.
        if data_in.dtype == 'float32':
            d_type = 'float32'
        else:
            d_type = 'float64'

        m_size = np.shape(mask)
        mask = mask.flatten()  # reshape mask to vector
        curr_size = [np.size(mask), *d_size[1:]]
        # pre-allocate new matrix
        data_out = np.full(tuple(curr_size), np.nan, dtype=d_type)
        data_out = np.reshape(data_out, (np.size(data_out, 0), -1))
        data_out[~mask, :] = np.reshape(data_in, (np.sum(~mask), -1))
        orig_size = [*m_size, *d_size[1:]]
        data_out = np.squeeze(np.reshape(data_out, tuple(orig_size)))

    return data_out

def mint_calc_score(data):
    '''
    This function calculates the loadings once so that the calc_score does not have to do so repeatedly.
    '''
    loadings = np.linalg.norm(data.svt, axis=0)**2

    def calc_score(y_true, y_pred):
        '''
        This function returns an R2 score based on the loadings of each component.
        '''
        
        numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        denominator = (
            (y_true - np.mean(y_true, axis=0)) ** 2
        ).sum(axis=0, dtype=np.float64)
        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([y_true.shape[1]])
        output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

        return np.average(output_scores, weights=loadings)

    return calc_score


def mint_score_func(design_df, data, cv=5, scoring='r2', n_jobs=-1):
    '''
    This function returns a function which calculates the negative weighted R2 score of a ridge model, 
    with the value of alpha being the only parameter. Used in conjungtion with fminbound.
    '''
    def score_func(alpha):
        '''
        Score function
        '''

        mdl = Ridge(alpha=alpha)
        pipeline = make_pipeline(StandardScaler(), mdl)
        scores = cross_val_score(pipeline, design_df, data,
                                 cv=cv, scoring=scoring, n_jobs=n_jobs)
        return -np.mean(scores)

    return score_func


def split_by_trials(data, opts):
    '''
    Make index that contain trials instead of random data points. 
    Based on Max Melin's ridge-model-matlab-baseline package.
    '''
    rng = np.random.default_rng(seed=4)
    trial_idx = rng.permutation(int(len(data) / opts['frames_per_trial']))
    frame_idx = np.arange(len(data))
    frame_idx = frame_idx.reshape(-1, opts['frames_per_trial'])
    frame_idx = frame_idx[trial_idx, :].flatten()
    out_sizes = np.full(opts['out_folds'], len(data) // opts['out_folds'], dtype=int)
    out_sizes[: len(data) % opts['out_folds']] += 1
    in_sizes = np.full(opts['in_folds'], (len(data) - out_sizes[0]) // opts['in_folds'], dtype=int)
    in_sizes[: (len(data) - out_sizes[0]) % opts['in_folds']] += 1
    out_split = []
    in_split = []
    out_current = 0
    for out_fold, out_size in enumerate(out_sizes):
        start, stop = out_current, out_current + out_size
        test_idx = frame_idx[start:stop]
        train_idx = np.append(frame_idx[:start], frame_idx[stop:])
        out_split.append((train_idx, test_idx))
        out_current = stop
        if out_fold == 0:
            frames_range = np.arange(len(data)-out_sizes[0])
            in_current = 0
            for in_size in in_sizes:
                start, stop = in_current, in_current + in_size
                test_idx = frames_range[start:stop]
                train_idx = np.append(frames_range[:start], frames_range[stop:])
                in_split.append((train_idx, test_idx))
                in_current = stop

    return out_split, in_split


def unstack(df):
    '''
    Function to unstack DataFrame while keeping order of rows/columns.
    '''
    new_index = df.index.droplevel(0).unique()
    new_cols = df.index.get_level_values(0).unique()

    return df.unstack(0).reindex(new_index).reindex(new_cols, level=-1, axis='columns')


def listify(object_):
    object_ = [object_] if not isinstance(object_, list) else object_ # turn string object into a list if it isn't already one
    return object_