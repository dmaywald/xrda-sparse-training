
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    
    temp = pd.DataFrame([[trials.trials[i]['result']['params'][param] for i in range(len(trials.trials))] for param in trials.trials[0]['result']['params'].keys()]).T
    temp.columns = list(trials.trials[0]['result']['params'].keys())
    
    for key in trials.trials[0]['result'].keys():
        if key == 'status' or key == 'params':
            continue        
        temp = temp.join(pd.Series([trials.trials[i]['result'][key] for i in range(len(trials.trials))],  name = key))


    temp['init_lr'] = 2 - temp['init_lr']
    
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