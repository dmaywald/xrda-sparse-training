# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:00:09 2024

@author: Owner
"""
from hyperopt import hp
import numpy as np
from scipy import stats

class ParameterSpace:

    def __init__(self, expected_lam = 1.3e-6, prob_max_lam = 5e-3, prob_max = .01,
                 init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
                 mom_ts = 9.5, b_mom_ts = 9.5):
        
        self.expected_lam = expected_lam
        self.prob_max_lam = prob_max_lam
        self.prob_max = prob_max
        self.init_lr_low = init_lr_low
        self.init_lr_high = init_lr_high
        self.av_low = av_low
        self.av_high = av_high
        self.mom_ts = mom_ts
        self.b_mom_ts = b_mom_ts
        
        
    def ResNetParamSpace(self):
        
        space = {
            'lam' : hp.lognormal('lam', np.log(self.expected_lam),
                                     (np.log(self.prob_max_lam/self.expected_lam)/stats.norm.ppf(1-self.prob_max))),
            'init_lr' : hp.loguniform('init_lr', self.init_lr_low, self.init_lr_high), # this is on a support of [1,2], remember to account for this in param call
            'av_param' : hp.uniform('av_param', self.av_low,self.av_high),
            'mom_ts' : hp.choice('mom_ts', [9.5]),
            'b_mom_ts' : hp.choice('b_mom_ts', [9.5]),
            'weight_decay' : hp.choice('weight_decay', [5e-4]) 
            }
        
        return space