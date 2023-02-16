'''
Module containing utility classes and functions.
'''

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


class SVDStack(object):
    '''
    This class created the image stack for widefield imaging data.
    '''

    def __init__(self, u, svt, dims=None, dtype='float32'):
        self.u = u.astype('float32')
        self.svt = svt.astype('float32')
        self.issparse = False
        if issparse(u):
            self.issparse = True
            if dims is None:
                raise ValueError(
                    'Supply dims = [H,W] when using sparse arrays')
            self.Uflat = self.u
        else:
            if dims is None:
                dims = u.shape[:2]
            self.Uflat = self.u.reshape(-1, self.u.shape[-1])
        self.shape = [svt.shape[1], *dims]
        self.dtype = dtype
        self.mask = np.isnan(u[:, :, 0])  # create the mask

    def __len__(self):
        return self.svt.shape[0]


def model_corr(r_stack, m_svt):
    '''
    Short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, u is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.

    Originally written in MATLAB by Simon Musall, 2019

    Adapted to Python by Michael Sokoletsky, 2021
    '''

    Vc = r_stack.svt.T
    Vm = m_svt.T

    if len(np.shape(r_stack.u)) == 3:
        u = array_shrink(r_stack.u, r_stack.mask)

    cov_Vc = np.cov(Vc)  # S x S
    cov_Vm = np.cov(Vm)  # % S x S
    c_cov_V = (Vm - np.expand_dims(np.mean(Vm, 1), axis=1)
               ) @ Vc.T / (np.size(Vc, 1) - 1)  # S x S
    cov_P = np.expand_dims(np.sum((u @ c_cov_V) * u, 1), axis=0)  # 1 x P
    var_P1 = np.expand_dims(np.sum((u @ cov_Vc) * u, 1), axis=0)  # 1 x Pii
    var_P2 = np.expand_dims(np.sum((u @ cov_Vm) * u, 1), axis=0)  # 1 x P
    std_Px_Py = var_P1 ** 0.5 * var_P2 ** 0.5  # 1 x P
    corr_mat = (cov_P / std_Px_Py).T
    corr_mat = array_shrink(corr_mat, r_stack.mask, 'split') ** 2

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


def mint_score_func(design_df, data, cv=5, scoring='r2', n_jobs=-1):
    '''
    This function returns a function which calculates the negative R2 score of a ridge model, 
    with the value of alpha being the only parameter. Used in conjungtion with fminbound.
    '''
    def score_func(alpha):
        '''
        Score function
        '''
        mdl = Ridge(alpha=alpha)
        scores = cross_val_score(mdl, design_df, data,
                                 cv=cv, scoring=scoring, n_jobs=n_jobs)
        return -np.mean(scores)

    return score_func


def split_by_trials(data, opts):
    '''
    Make index that contain trials instead of random data points. 
    Based on Max Melin's ridge-model-matlab-baseline package.
    '''
    rng = np.random.default_rng(seed=4)
    trial_idx = rng.permutation(int(len(data.svt) / opts['frames_per_trial']))
    frame_idx = np.arange(len(data))
    frame_idx = frame_idx.reshape(-1, opts['frames_per_trial'])
    frame_idx = frame_idx[trial_idx, :].flatten()
    outer_size = int(np.floor(len(data)/opts['out_folds']))
    inner_size = int(np.floor((len(data)-outer_size)/opts['in_folds']))

    out_split = []
    in_split = []
    for out_fold in range(opts['out_folds']):
        test_idx = frame_idx[out_fold*outer_size:(out_fold+1)*outer_size]
        train_idx = np.append(frame_idx[:out_fold*outer_size], frame_idx[(out_fold+1)*outer_size:])
        out_split.append((train_idx, test_idx))
        if out_fold == 0:
            for in_fold in range(opts['in_folds']):
                frames_range = np.arange(len(data)-outer_size)
                test_idx = frames_range[in_fold*inner_size:(in_fold+1)*inner_size]
                train_idx = np.append(frames_range[:in_fold*inner_size], frames_range[(in_fold+1)*inner_size:])
                in_split.append((train_idx, test_idx))

    return out_split, in_split


def unstack(df):
    '''
    Function to unstack DataFrame while keeping order of rows/columns.
    '''
    new_index = df.index.droplevel(0).unique()
    new_cols = df.index.get_level_values(0).unique()

    return df.unstack(0).reindex(new_index).reindex(new_cols, level=-1, axis='columns')
