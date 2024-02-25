# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:00:09 2024

@author: Owner
"""
from hyperopt import hp
import numpy as np
import scipy.stats as stat

                
def ResNetParamSpace(expected_lam = 1.3e-6, prob_max_lam = 5e-3, prob_max = .01,
             init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
             mom_ts = 9.5, b_mom_ts = 9.5):
    
    space = {
        'lam' : hp.lognormal('lam', np.log(expected_lam),
                                 (np.log(prob_max_lam/expected_lam)/stat.norm.ppf(1-prob_max))),
        'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
        'av_param' : hp.uniform('av_param', av_low, av_high),
        'mom_ts' : hp.choice('mom_ts', [9.5]),
        'b_mom_ts' : hp.choice('b_mom_ts', [9.5]),
        'weight_decay' : hp.choice('weight_decay', [5e-4]) 
        }
    
    return space



def VggParamSpace(expected_lam = 1e-6, prob_max_lam = 5e-3, prob_max = .01,
             init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
             mom_ts = 9.5, b_mom_ts = 9.5):
    
    space = {
        'lam' : hp.lognormal('lam', np.log(expected_lam),
                                 (np.log(prob_max_lam/expected_lam)/stat.norm.ppf(1-prob_max))),
        'init_lr' : hp.loguniform('init_lr', init_lr_low, init_lr_high), # this is on a support of [1,2], remember to account for this in param call
        'av_param' : hp.uniform('av_param', av_low, av_high),
        'mom_ts' : hp.choice('mom_ts', [9.5]),
        'b_mom_ts' : hp.choice('b_mom_ts', [9.5]),
        'weight_decay' : hp.choice('weight_decay', [5e-4]) 
        }
    
    return space