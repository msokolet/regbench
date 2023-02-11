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

import numpy as np 
from tqdm.notebook import tqdm
from scipy.sparse import issparse


from .ridge import ridge_MML

def reconstruct(u, svt, dims = None):
    '''
    This method reconstructs the full widefield map from the reduced dimensions.
    '''
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]
    return u.dot(svt).reshape((*dims,-1)).transpose(-1,0,1).squeeze()
    
    
class SVDStack(object):
    '''
    This class created the image stack for widefield imaging data.
    '''
    def __init__(self, U, SVT, dims = None, dtype = 'float32'):
        self.U = U.astype('float32')
        self.SVT = SVT.astype('float32')
        self.issparse = False
        
        if issparse(U):
            self.issparse = True
            if dims is None:
                raise ValueError('Supply dims = [H,W] when using sparse arrays')
            self.Uflat = self.U
        else:
            if dims is None:
                dims = U.shape[:2]
            self.Uflat = self.U.reshape(-1,self.U.shape[-1])
        self.shape = [SVT.shape[1],*dims]
        self.dtype = dtype
        self.mask = np.isnan(U[:,:,0]) # create the mask
        
            
    def split(self, folds):
        '''
        Split
        '''
        if folds==1:
            print('Ridgefold is <= 1, fit to complete dataset instead')
            
        rng = np.random.default_rng(1) # for reproducibility
        rand_idx = rng.permutation(len(self)) # generate randum number index for splitting training and testing
        fold_cnt = np.floor(len(self) / folds).astype(np.uint)       
        
        for i_fold in range(folds):
            train_idx = np.ones(len(self),dtype=np.bool_)
            train_idx[rand_idx[(i_fold*fold_cnt) + np.arange(fold_cnt)]] = False # indexes for training data

            yield i_fold, train_idx # yield successive training folds and their indices
        
    def train(self, train_idx, cR, c_ridge = None, suppress_output = False):
        '''
        Train
        '''
        return ridge_MML(self.SVT[:,train_idx].T, cR.loc[train_idx], recenter = True, L = c_ridge, display_failures = not suppress_output)
    
    def test(self, train_idx, cR, c_beta):
        '''
        Test
        '''
        self.SVT[:, ~train_idx] = (cR.loc[~train_idx] @ c_beta).T
        
    def __len__(self):
        return self.SVT.shape[1]
    
    def __getitem__(self,*args):
        ndims  = len(args)
        if type(args[0]) is slice:
            idxz = range(*args[0].indices(self.shape[0]))
        else:
            idxz = args[0]        
        return reconstruct(self.U,self.SVT[:,idxz],dims = self.shape[1:])
  
    
    
def model_corr(r_stack, m_stack):
    
    """
    short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, U is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.
    
    Originally written in MATLAB by Simon Musall, 2019
    
    Adapted to Python by Michael Sokoletsky, 2021
    """
    
    Vc = np.reshape(r_stack.SVT,(np.size(r_stack.SVT,0),-1))
    Vm = np.reshape(m_stack.SVT,(np.size(m_stack.SVT,0),-1))
    if len(np.shape(r_stack.U)) == 3:
        U = array_shrink(r_stack.U, r_stack.mask)

    cov_Vc = np.cov(Vc) # S x S
    cov_Vm = np.cov(Vm) # % S x S
    c_cov_V = (Vm - np.expand_dims(np.mean(Vm,1), axis=1)) @ Vc.T / (np.size(Vc, 1) - 1) # S x S
    cov_P = np.expand_dims(np.sum((U @ c_cov_V) * U, 1),axis=0) # 1 x P
    var_P1 = np.expand_dims(np.sum((U @ cov_Vc) * U, 1),axis=0) # 1 x Pii
    var_P2 = np.expand_dims(np.sum((U @ cov_Vm) * U, 1),axis=0) # 1 x P
    std_Px_Py = var_P1 ** 0.5 * var_P2 ** 0.5 # 1 x P
    corr_mat = ((cov_P / std_Px_Py).T)**2
    corr_mat = array_shrink(corr_mat, r_stack.mask,'split')
    
    return corr_mat

def compute_R2(r_stack, m_stack):
    
    """
    short code to compute the correlation between lowD data Vc and modeled
    lowD data Vm. Vc and Vm are temporal components, U is the spatial
    components. corr_mat is a the correlation between Vc and Vm in each pixel.
    
    Originally written in MATLAB by Simon Musall, 2019
    
    Adapted to Python by Michael Sokoletsky, 2021
    """
    
    Vc = np.reshape(r_stack.SVT,(np.size(r_stack.SVT,0),-1))
    Vm = np.reshape(m_stack.SVT,(np.size(m_stack.SVT,0),-1))
    if len(np.shape(r_stack.U)) == 3:
        U = array_shrink(r_stack.U, np.squeeze(np.isnan(r_stack.U[:,:,0])))

    cov_Vc = np.cov(Vc) # S x S
    cov_Vm = np.cov(Vm) # % S x S
    c_cov_V = (Vm - np.expand_dims(np.mean(Vm,1), axis=1)) @ Vc.T / (np.size(Vc, 1) - 1) # S x S
    cov_P = np.expand_dims(np.sum((U @ c_cov_V) * U, 1),axis=0) # 1 x P
    var_P1 = np.expand_dims(np.sum((U @ cov_Vc) * U, 1),axis=0) # 1 x Pii
    var_P2 = np.expand_dims(np.sum((U @ cov_Vm) * U, 1),axis=0) # 1 x P
    std_Px_Py = var_P1 ** 0.5 * var_P2 ** 0.5 # 1 x P
    corr_mat = ((cov_P / std_Px_Py).T)**2
    corr_mat = array_shrink(corr_mat, r_stack.mask,'split')
    
    return corr_mat, var_P1, var_P2


def array_shrink(data_in, mask, mode='merge'):

    """
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

    """

    d_size = np.shape(data_in) # size of input matrix
    if d_size[0] == 1:
        data_in = np.squeeze(data_in) # remove singleton dimensions
        d_size = np.shape(data_in)

    if len(d_size) == 2:
        if d_size[0] == 1:
            data_in = data_in.T
            d_size = np.shape(data_in) # size of input matrix

        d_size = d_size + (1,)

    if mode == 'merge': # merge x and y dimension into one
    
        data_in = np.reshape(data_in,(np.size(mask),np.prod(d_size[mask.ndim:]))) # merge x and y dimension based on mask size and remaining dimensions.
        mask = mask.flatten() # reshape mask to vector
        data_in = data_in[~mask,:]
        orig_size = [np.size(data_in,0),*d_size[2:]]
        data_in = np.reshape(data_in,tuple(orig_size))
        data_out =  data_in

    elif mode == 'split': # split first dimension into x- and y- dimension based on mask size

        # check if datatype is single. If not will use double as a default.
        if data_in.dtype == 'float32':
            d_type = 'float32'
        else:
            d_type = 'float64'

        m_size = np.shape(mask)
        mask = mask.flatten() # reshape mask to vector
        curr_size = [np.size(mask), *d_size[1:]]
        data_out = np.full(tuple(curr_size),np.nan,dtype=d_type) # pre-allocate new matrix
        data_out = np.reshape(data_out,(np.size(data_out,0),-1))
        data_out[~mask,:] = np.reshape(data_in,(np.sum(~mask),-1))
        orig_size = [*m_size, *d_size[1:]]
        data_out = np.squeeze(np.reshape(data_out,tuple(orig_size)))

    return data_out

def cross_val_model(data, design_df, opts, method = 'MMLSimon'):

    '''
    This function computes the cross-validated R^2.
    '''

    # labels_idx = np.nonzero(np.isin(reg_labels, c_labels))
    # c_idx = np.isin(reg_idx,labels_idx) # get index for regressors
    # c_labels = reg_labels[np.sort(labels_idx)] # make sure c_labels is in the right order
    
    # # create new regressor index that matches c labels
    # sub_idx = copy.copy(reg_idx)
    # sub_idx = sub_idx[c_idx]
    # temp = np.unique(sub_idx)
    # for x, x_idx in enumerate(temp):
    #     sub_idx[sub_idx == x] = x_idx

    m_stack = SVDStack(data.U, np.zeros_like(data.SVT)) # pre-allocate modeled stack
    c_beta = [0]*opts['folds']
    folds_gen = data.split(opts['folds']) # split the real stack into folds for training the model
    
    if method == 'mml_simon':
        for i_fold, train_idx in tqdm(folds_gen, desc = 'Fold number', total=opts['folds']):
            if i_fold == 0:
                c_ridge, c_beta[i_fold] = data.train(train_idx, design_df) # train the model on training indexes in current fold
            else:
                c_beta[i_fold] = data.train(train_idx, design_df, c_ridge) # train the model on training indexes in current fold. ridge value should be the same as in the first run.
            m_stack.test(train_idx, design_df, c_beta[i_fold]) # apply the model on the remaining (testing) indexes in the modeled stack
    elif method == 'mml_sklearn':



        
    return m_stack, c_ridge


    