from hyperopt import hp
import numpy as np
import scipy.stats as stat

                
def ResNetParamSpace(expected_lam = 1.3e-6, max_lam = 5e-3, prob_max_lam = .01,
             init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
             mom_ts = 9.5, b_mom_ts = 9.5, sigma_mom_ts = 1, sigma_b_mom_ts = 1,
             expected_wd = 1e-3, max_wd = 1e-2, prob_max_wd = .01):
    """
    

    Parameters
    ----------
    expected_lam : float, optional
        Expected value of lognormal distribution defining the lambda hyperparameter. The default is 1.3e-6.
    max_lam : float, optional
        Tail end value of lambda within lognormal distribution. Needs to be larger than expected_lam. Used with 
        prob_max_lam to determine the tail shape of the lognormal distribution of lambda. The default is 5e-3.
    prob_max_lam : TYPE, optional
        The probability of lambda being larger than max_lam when drawing from lognormal distribution. The default is .01.
    init_lr_low : float, optional
        lower bound of log uniform distribution that describes init_lr. The default is 0 (i.e. log(1)).
    init_lr_high : float, optional
        upper bound of log uniform distribution that describes init_lr. The default is np.log(2).
    av_low : float, optional
        lower bound of uniform distibution that describes av_param. The default is 0.
    av_high : float, optional
        upper bound of uniform distribution that describes av_param. The default is 1.
    mom_ts : float, optional
        mean of normal distribution describing mom_ts. The default is 9.5.
    b_mom_ts : float, optional
        mean of normal distribution that describes b_mom_ts. The default is 9.5.
    sigma_mom_ts : float, optional
        standard deviation of normal distribution that describes mom_ts. The default is 1.
    sigma_b_mom_ts : float, optional
        standard deviation of normal distribution that describes b_mom_ts. The default is 1.
    expected_wd : float, optional
        Expected value of lognormal distribution defining the weight_decay hyperparameter. Set to None to not tune this
        parameter. The default is 1e-3.
    max_wd : float, optional
        If expected_wd is not None, then this is the tail end value of weight_decay within lognormal distribution.
        Needs to be larger than expected_wd. Used with prob_max_wd to determine the tail shape of the lognormal distribution of lambda.  
        The default is 1e-2.
    prob_max_wd : float, optional
        If expected wd_ is not None, then this is the probability of weight_decay being
        larger than max_wd when drawing from lognormal distribution. The default is .01.

    Returns
    -------
    space : dict
        Dictionary of hyperopt parameter spaces used in bayesian optimization.
        See https://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for more

    """
    if expected_wd is not None:
        space = {
            'lam' : hp.lognormal('lam', np.log(expected_lam),
                                     (np.log(max_lam/expected_lam)/stat.norm.ppf(1-prob_max_lam))),
            'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', av_low, av_high),
            'mom_ts' : hp.normal('mom_ts', mom_ts, sigma_mom_ts),
            'b_mom_ts' : hp.normal('b_mom_ts', b_mom_ts, sigma_b_mom_ts),
            'weight_decay' : hp.lognormal('weight_decay', np.log(expected_wd),
                                     (np.log(max_wd/expected_wd)/stat.norm.ppf(1-prob_max_wd)))
            }
    if expected_wd is None: 
        space = {
            'lam' : hp.lognormal('lam', np.log(expected_lam),
                                     (np.log(max_lam/expected_lam)/stat.norm.ppf(1-prob_max_lam))),
            'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', av_low, av_high),
            'mom_ts' : hp.normal('mom_ts', mom_ts, sigma_mom_ts),
            'b_mom_ts' : hp.normal('b_mom_ts', b_mom_ts, sigma_b_mom_ts),
            'weight_decay' : hp.choice('weight_decay', [5e-4])
            }
    return space



def VggParamSpace(expected_lam = 1e-6, max_lam = 5e-3, prob_max_lam = .01,
             init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
             mom_ts = 9.5, b_mom_ts = 9.5, sigma_mom_ts = 1, sigma_b_mom_ts = 1,
             expected_wd = 5e-4, max_wd = 1e-2, prob_max_wd = .01):
    """
    

    Parameters
    ----------
    expected_lam : float, optional
        Expected value of lognormal distribution defining the lambda hyperparameter. The default is 1e-6.
    max_lam : float, optional
        Tail end value of lambda within lognormal distribution. Needs to be larger than expected_lam. Used with 
        prob_max_lam to determine the tail shape of the lognormal distribution of lambda. The default is 5e-3.
    prob_max_lam : float, optional
        The probability of lambda being larger than max_lam when drawing from lognormal distribution. The default is .01.
    init_lr_low : float, optional
        lower bound of log uniform distribution that describes init_lr. The default is 0 (i.e. log(1)).
    init_lr_high : float, optional
        upper bound of log uniform distribution that describes init_lr. The default is np.log(2).
    av_low : float, optional
        lower bound of uniform distibution that describes av_param. The default is 0.
    av_high : float, optional
        upper bound of uniform distribution that describes av_param. The default is 1.
    mom_ts : float, optional
        mean of normal distribution describing mom_ts. The default is 9.5.
    b_mom_ts : float, optional
        mean of normal distribution that describes b_mom_ts. The default is 9.5.
    sigma_mom_ts : float, optional
        standard deviation of normal distribution that describes mom_ts. The default is 1.
    sigma_b_mom_ts : float, optional
        standard deviation of normal distribution that describes b_mom_ts. The default is 1.
    expected_wd : float, optional
        Expected value of lognormal distribution defining the weight_decay hyperparameter. Set to None to not tune this
        parameter. The default is 5e-4.
    max_wd : float, optional
        If expected_wd is not None, then this is the tail end value of weight_decay within lognormal distribution.
        Needs to be larger than expected_wd. Used with prob_max_wd to determine the tail shape of the lognormal distribution of lambda.  
        The default is 1e-2.
    prob_max_wd : float, optional
        If expected wd_ is not None, then this is the probability of weight_decay being
        larger than max_wd when drawing from lognormal distribution. The default is .01.

    Returns
    -------
    space : dict
        Dictionary of hyperopt parameter spaces used in bayesian optimization.
        See https://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for more

    """
    if expected_wd is not None:
            
        space = {
            'lam' : hp.lognormal('lam', np.log(expected_lam),
                                     (np.log(max_lam/expected_lam)/stat.norm.ppf(1-prob_max_lam))),
            'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', av_low, av_high),
            'mom_ts' : hp.normal('mom_ts', mom_ts, sigma_mom_ts),
            'b_mom_ts' : hp.normal('b_mom_ts', b_mom_ts, sigma_b_mom_ts),
            'weight_decay' : hp.lognormal('weight_decay', np.log(expected_wd),
                                     (np.log(max_wd/expected_wd)/stat.norm.ppf(1-prob_max_wd)))
            }
    if expected_wd is None: 
        space = {
            'lam' : hp.lognormal('lam', np.log(expected_lam),
                                     (np.log(max_lam/expected_lam)/stat.norm.ppf(1-prob_max_lam))),
            'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', av_low, av_high),
            'mom_ts' : hp.normal('mom_ts', mom_ts, sigma_mom_ts),
            'b_mom_ts' : hp.normal('b_mom_ts', b_mom_ts, sigma_b_mom_ts),
            'weight_decay' : hp.choice('weight_decay', [5e-4])
            }
    
    return space


def DenseNetParamSpace(expected_lam = 1e-6, max_lam = 5e-3, prob_max_lam = .01,
             init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
             mom_ts = 9.5, b_mom_ts = 9.5, sigma_mom_ts = 1, sigma_b_mom_ts = 1,
             expected_wd = 5e-4, max_wd = 1e-2, prob_max_wd = .01):
    """
    

    Parameters
    ----------
    expected_lam : float, optional
        Expected value of lognormal distribution defining the lambda hyperparameter. The default is 1e-6.
    max_lam : float, optional
        Tail end value of lambda within lognormal distribution. Needs to be larger than expected_lam. Used with 
        prob_max_lam to determine the tail shape of the lognormal distribution of lambda. The default is 5e-3.
    prob_max_lam : TYPE, optional
        The probability of lambda being larger than max_lam when drawing from lognormal distribution. The default is .01.
    init_lr_low : float, optional
        lower bound of log uniform distribution that describes init_lr. The default is 0 (i.e. log(1)).
    init_lr_high : float, optional
        upper bound of log uniform distribution that describes init_lr. The default is np.log(2).
    av_low : float, optional
        lower bound of uniform distibution that describes av_param. The default is 0.
    av_high : float, optional
        upper bound of uniform distribution that describes av_param. The default is 1.
    mom_ts : float, optional
        mean of normal distribution describing mom_ts. The default is 9.5.
    b_mom_ts : float, optional
        mean of normal distribution that describes b_mom_ts. The default is 9.5.
    sigma_mom_ts : float, optional
        standard deviation of normal distribution that describes mom_ts. The default is 1.
    sigma_b_mom_ts : float, optional
        standard deviation of normal distribution that describes b_mom_ts. The default is 1.
    expected_wd : float, optional
        Expected value of lognormal distribution defining the weight_decay hyperparameter. Set to None to not tune this
        parameter. The default is 5e-4.
    max_wd : float, optional
        If expected_wd is not None, then this is the tail end value of weight_decay within lognormal distribution.
        Needs to be larger than expected_wd. Used with prob_max_wd to determine the tail shape of the lognormal distribution of lambda.  
        The default is 1e-2.
    prob_max_wd : float, optional
        If expected wd_ is not None, then this is the probability of weight_decay being
        larger than max_wd when drawing from lognormal distribution. The default is .01.

    Returns
    -------
    space : dict
        Dictionary of hyperopt parameter spaces used in bayesian optimization.
        See https://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for more

    """
    if expected_wd is not None:
        space = {
            'lam' : hp.lognormal('lam', np.log(expected_lam),
                                     (np.log(max_lam/expected_lam)/stat.norm.ppf(1-prob_max_lam))),
            'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', av_low, av_high),
            'mom_ts' : hp.normal('mom_ts', mom_ts, sigma_mom_ts),
            'b_mom_ts' : hp.normal('b_mom_ts', b_mom_ts, sigma_b_mom_ts),
            'weight_decay' : hp.lognormal('weight_decay', np.log(expected_wd),
                                     (np.log(max_wd/expected_wd)/stat.norm.ppf(1-prob_max_wd)))
            }
    if expected_wd is None: 
        space = {
            'lam' : hp.lognormal('lam', np.log(expected_lam),
                                     (np.log(max_lam/expected_lam)/stat.norm.ppf(1-prob_max_lam))),
            'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', av_low, av_high),
            'mom_ts' : hp.normal('mom_ts', mom_ts, sigma_mom_ts),
            'b_mom_ts' : hp.normal('b_mom_ts', b_mom_ts, sigma_b_mom_ts),
            'weight_decay' : hp.choice('weight_decay', [5e-4])
            }
    return space