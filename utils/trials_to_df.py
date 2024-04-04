# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:12:06 2024

@author: Devon
"""
import os
import pandas as pd

def trials_to_df(trials, save_trials_str = None):
    """
    

    Parameters
    ----------
    trials : bayes trials object from hyperopt
        Trials dictionary of information gathered from bayesian optimization

    save_trials_str : str
        String indicating where to save generated dataframe as csv
    Returns
    -------
    Data frame of results from bayes trials object
    """
    
    temp = pd.DataFrame([[trials.trials[i]['result']['params'][param] for i in range(len(trials.trials))] for param in ['init_lr', 'lam', 'av_param', 'mom_ts', 'b_mom_ts', 'weight_decay']]).T
    temp.columns = ['init_lr', 'lam', 'av_param', 'mom_ts', 'b_mom_ts', 'weight_decay']
    
    temp = temp.join(pd.Series([trials.trials[i]['result']['loss'] for i in range(len(trials.trials))],  name = 'loss'))

    temp.init_lr = 2 - temp.init_lr
    
    if save_trials_str is not None:
        str_list = save_trials_str.split("/")[:-1]
        if len(str_list) > 0: 
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
    if save_trials_str is not None:
        temp.to_csv(save_trials_str, index = True)
    
    
    return temp