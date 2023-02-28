'''
Docstring
'''

import numpy as np
from scipy import optimize

def ridge_MML(Y, X, recenter = True, L = None, regress = True):
    """
    This is an implementation of Ridge regression with the Ridge parameter
    lambda determined using the fast algorithm of Karabatsos 2017 (see
    below). I also made some improvements, described below.

    Inputs are Y (the outcome variables) and X (the design matrix, aka the
    regressors). Y may be a matrix. X is a matrix with as many rows as Y, and
    should *not* include a column of ones.

    A separate value of lambda will be found for each column of Y.

    Outputs are the lambdas (the Ridge parameters, one per column of Y); the
    betas (the regression coefficients, again with columns corresponding to
    columns of Y); and a vector of logicals telling you whether fminbnd
    failed to converge for each column of y (this happens frequently).

    If recenter is True (default), the columns of X and Y will be recentered at
    0. betas will be of size:  np.size(X, 1) x np.size(Y, 1)
    To reconstruct the recentered Y, use:
    y_recentered_hat = (X - np.mean(X, 0)) * betas

    If recenter is False, the columns of X and Y will not be recentered. betas
    will be of size:  size(X, 2)+1 x size(Y, 2)
    The first row corresponds to the intercept term.
    To reconstruct Y, you may therefore use either:
    y_hat = [np.ones((np.size(X, 0), 1)), X] * betas
    or
    y_hat = X * betas[1:, :] + betas[0, :]

    If lambdas is supplied, the optimization step is skipped and the betas
    are computed immediately. This obviously speeds things up a lot.


    TECHNICAL DETAILS:

    To allow for un-centered X and Y, it turns out you can simply avoid
    penalizing the intercept when performing the regression. However, no
    matter what it is important to put the columns of X on the same scale
    before performing the regression (though Matlab's ridge.m does not do
    this if you choose not to recenter). This rescaling step is undone in the
    betas that are returned, so the user does not have to worry about it. But
    this step substantially improves reconstruction quality.

    Improvements to the Karabatsos algorithm: as lambda gets large, local
    optima occur frequently. To combat this, I use two strategies. First,
    once we've passed lambda = 25, we stop using the fixed step size of 1/4
    and start using an adaptive step size: 1% of the current lambda. (This
    also speeds up execution time greatly for large lambdas.) Second, we add
    boxcar smoothing to our likelihood values, with a width of 7. We still
    end up hitting local minima, but the lambdas we find are much bigger and
    closer to the global optimum.

    Source: "Marginal maximum likelihood estimation methods for the tuning
    parameters of ridge, power ridge, and generalized ridge regression" by G
    Karabatsos, Communications in Statistics -- Simulation and Computation,
    2017. Page 6.
    http://www.tandfonline.com/doi/pdf/10.1080/03610918.2017.1321119

    Written by Matt Kaufman, 2018. mattkaufman@uchicago.edu
    
    Adapted to Python by Michael Sokoletsky, 2021
    """
    
    ## Optional arguments

    if L is None:
        compute_L = True
    else:
        compute_L = False

    ## If design matrix is a DataFrame, convert to a matrix

    X = np.array(X)

    ## Error checking

    if np.size(Y, 0) != np.size(X, 0):
        raise IndexError('Size mismatch')

    ## Ensure Y is zero-mean
    # This is needed to estimate lambdas, but if recenter = 0, the mean will be
    # restored later for the beta estimation

    pY = np.size(Y, 1)

    if compute_L or recenter:
        X[np.isnan(X)] = 0

    ## Optimize lambda

    if compute_L:

        ## SVD the predictors

        U, d, VH = np.linalg.svd(X, full_matrices=False)
        S = np.diag(d)
        V = VH.T.conj()

        ## Find the valid singular values of X, compute d and alpha

        n = np.size(X, 0)  # Observations
        p = np.size(V, 1)  # Predictors

        # Find the number of good singular values. Ensure numerical stability.
        q = np.sum(d.T > abs(np.spacing(U[0,0])) * np.arange(1,p+1))

        d2 = d ** 2

        # Equation 1
        # Eliminated the diag(1 ./ d2) term: it gets cancelled later and only adds
        # numerical instability (since later elements of d may be tiny).
        # alph = V' * X' * Y
        alph = S @ U.T @ Y
        alpha2 = alph ** 2

        ## Compute variance of y
        # In Equation 19, this is shown as y'y

        Y_var = np.sum(Y ** 2, 0)

        ## Compute the lambdasnp.

        L = np.full(pY,np.nan)

        convergence_failures = np.empty(pY, dtype=int)
        
        for i in range(pY):
            
            L[i], flag = ridge_MML_one_Y(q, d2, n, Y_var[i], alpha2[:, i])
            convergence_failures[i] = flag
        
    else:
        p = np.size(X, 1)




    # If requested, perform the actual regression

    if regress:


        betas = np.full((p, pY), np.nan)

        # You would think you could compute X'X more efficiently as VSSV', but
        # this is numerically unstable and can alter results slightly. Oh well.
        # XTX = V * bsxfun(@times, V', d2)

        XTX = X.T @ X

        # Prep penalty matrix
        ep = np.identity(p)


        # Compute X' * Y all at once, again for speed
        XTY = X.T @ Y

        # Compute betas for renormed X
        if hasattr(L, "__len__"):
            for i in range(0, pY):
                betas[:, i] = np.linalg.solve(XTX + L[i] * ep, XTY[:, i])
        else:
            betas = np.linalg.solve(XTX + L * ep, XTY)

        betas[np.isnan(betas)] = 0



    ## Display fminbnd failures
    
    if compute_L and sum(convergence_failures) > 0:
        print(f'fminbnd failed to converge {sum(convergence_failures)}/{pY} times')
    
    if compute_L and regress:
        return L, betas
    if compute_L:
        return L
    return betas
    

def ridge_MML_one_Y(q, d2, n, Y_var, alpha2):
    
    # Compute the lambda for one column of Y

    # Width of smoothing kernel to use when dealing with large lambda
    
    smooth = 7

    # Value of lambda at which to switch from step size 1/4 to step size L/stepDenom.
    # Value of stepSwitch must be >= smooth/4, and stepSwitch/stepDenom should
    # be >= 1/4.
    step_switch = 25
    step_denom = 100
    
    ## Set up smoothing

    # These rolling buffers will hold the last few values, to average for smoothing
    sm_buffer = np.full(smooth, np.nan)
    test_vals_L = np.full(smooth, np.nan)

    # Initialize index of the buffers where we'll write the next value
    sm_buffer_I = 0
                
    
    # Evaluate the log likelihood of the data for increasing values of lambda
    # This is step 1 of the two-step algorithm at the bottom of page 6.
    # Basically, increment until we pass the peak. Here, I've added trying
    # small steps as normal, then switching over to using larger steps and
    # smoothing to combat local minima.
    
    ## Mint the negative log-likelihood function
    NLL_func = mint_NLL_func(q, d2, n, Y_var, alpha2)



    # Loop through first few values of k before you apply smoothing.
    # Step size 1/4, as recommended by Karabatsos

    done = False
    NLL = np.inf
    for k in range(step_switch * 4+1):
        sm_buffer_I = sm_buffer_I % smooth +1
        prev_NLL = NLL

      # Compute negative log likelihood of the data for this value of lambda
        NLL = NLL_func(k / 4)
        
      # Add to smoothing buffer
        sm_buffer[int(sm_buffer_I-1)] = NLL
        test_vals_L[int(sm_buffer_I-1)] = k / 4

      # Check if we've passed the minimum
        if NLL > prev_NLL:
            # Compute limits for L
            min_L = (k - 2) / 4
            max_L = k / 4
            done = True
            break
                        
    # If we haven't already hit the max likelihood, continue increasing lambda,
    # but now apply smoothing to try to reduce the impact of local minima that
    # occur when lambda is large

    # Also increase step size from 1/4 to L/stepDenom, for speed and robustness
    # to local minima
    
    if not done:
        
        L = k / 4
        NLL = np.mean(sm_buffer)
        iteration = 0
        
        while not done:
            L += L / step_denom
            sm_buffer_I = sm_buffer_I % smooth + 1
            prev_NLL = NLL
            iteration += 1
            # Compute negative log likelihood of the data for this value of lambda,
            # overwrite oldest value in the smoothing buffer
            sm_buffer[int(sm_buffer_I-1)] = NLL_func(L)
            if (L + d2[:q]) == 0:
                pass
            test_vals_L[int(sm_buffer_I-1)] = L
            NLL = np.mean(sm_buffer)
            
            # Check if we've passed the minimum or hit NaN NLL (L passed double-precision maximum)
            
            if NLL>prev_NLL:
                # Adjust for smoothing kernel (walk back by half the kernel)
                sm_buffer_I -= (smooth - 1) / 2
                sm_buffer_I += smooth * (sm_buffer_I < 0) # wrap around
                
        
                max_L = test_vals_L[int(sm_buffer_I-1)]
            
                # Walk back by two more steps to find min bound
                sm_buffer_I -= 2
                sm_buffer_I += smooth * (sm_buffer_I < 0) # wrap around
                min_L = test_vals_L[int(sm_buffer_I-1)]

                passed_min = True
                done = True

            elif np.isnan(NLL):

                passed_min = False
                done = True
                
    else:

        passed_min = True

 
    ## Bounded optimization of lambda
    # This is step 2 of the two-step algorithm at the bottom of page 6. Note
    # that Karabatsos made a mistake when describing the indexing relative to
    # k*, which is fixed here (we need to go from k*-2 to k*, not k*-1 to k*+1)

    if passed_min:
        L, _, flag, _ = optimize.fminbound(NLL_func, max(0, min_L), max_L, xtol=1e-04, full_output=1, disp=0)
    else:
        flag = 1 # if the above loop could not find the minimum, return failed-to-converge flag
    
    return L, flag


def  mint_NLL_func(q, d2, n, Y_var, alpha2):
    '''
    Mint an anonymous function with L as the only input parameter, with all
    the other terms determined by the data.
    We've modified the math here to eliminate the d^2 term from both alpha
    (Equation 1, in main function) and here (Equation 19), because they
    cancel out and add numerical instability.
    '''
    NLL_func = lambda L: - (q * np.log(L) - np.sum(np.log(L + d2[:q])) \
                - n * np.log(Y_var - np.sum( np.divide(alpha2[:q],(L + d2[:q])))))
    return NLL_func
