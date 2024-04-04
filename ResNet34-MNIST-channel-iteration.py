import time
import numpy as np
import math

import torch
import torchvision.transforms as transforms

from models import mnist_resnet34
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import IterationSpecs
from utils import progress_dataframe
from utils import trials_to_df

from hyperparameter_optimization import tune_parameters
from hyperparameter_optimization import ResNetParamSpace


if __name__ == '__main__':
    t0 = time.time()

    init_params = {
        'init_lr': 1.0, #1.0 by default
        'lam': 1e-8, #1e-8 by default
        'av_param': 0.0, # 0.0 by default
        'mom_ts': 9.5, # 9.5 by default
        'b_mom_ts': 9.5, # 9.5 by default
        'weight_decay': 5e-4 # 5e-4 by default
        }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = mnist_resnet34().to(device)
    
    train_batch_size = 128
    num_epoch = 30
    subset_Data = None
    mode = 'channel' # normal, channel, or kernel
    save_str = 'mnist_resnet34_channel_iteration_specs'
    
    
    transform_train = transforms.Compose(
        [transforms.RandomCrop(28, padding=4),
          # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
          transforms.ToTensor()])
    
    transform_val = transforms.Compose([transforms.ToTensor()])
    
    training_specs = IterationSpecs(
        step_size = 2 - init_params['init_lr'], 
        mom_ts = init_params['mom_ts'], 
        b_mom_ts = init_params['b_mom_ts'], 
        weight_decay = init_params['weight_decay'], 
        av_param = init_params['av_param'])
    
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500, mode= mode))
    
    
    model_output_file = 'results/model_data/resnet/'+save_str+'_sparse_model_init_params.dat'
    progress_data_output_file = 'results/progress_data/resnet/'+save_str+'_sparse_training_progress_init_params.csv'
    
    # model_output_file = None
    # progress_data_output_file = None
    
    # Use the following below to visualize necessary 'num_epoch' for bayesian optimization
    progress_df = progress_dataframe(model=model,
                                     params=init_params,
                                     model_output_file=model_output_file,
                                     progress_data_output_file = progress_data_output_file,
                                     data = 'mnist',
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     optimizer = optimizer,
                                     training_specs = training_specs,
                                     train_batch_size=128,
                                     subset_Data= subset_Data,
                                     num_epoch= num_epoch,
                                     # At about every 3 epochs, halve step size and double averaging.
                                     epoch_updates= [6,9,12.15,18,21,24,27,30])

    # From the dataframe above:
    #     Training accuracy reaches maximum 99% after 5 epochs on the full
    #     Testing accuracy reaches 97% after 5 epochs and 99% after 10 epochs
    #     Sparsity is 84% after 10 epochs and 92% after 20 epochs
    #     Cross Entropy Loss is minimized to [.004, .16] after 2 epochs
    

    params_output_file = 'results/bayes_opt_params/resnet/'+save_str+'_sparse_model_bayes_params_small_data.dat'
    trials_output_file = 'results/bayes_opt_params/resnet/'+save_str+'_sparse_model_bayes_trials_small_data.dat'
    trials_df_output_file = 'results/bayes_opt_params/resnet/'+save_str+'_sparse_model_bayes_trials_small_data.csv'
    
    # params_output_file = None
    # trials_output_file = None
    
    model = mnist_resnet34().to(device)
    
    # experimentally determined 5e-6 is too large for lambda
    space = ResNetParamSpace(expected_lam = 5e-7, max_lam = 5e-6, prob_max_lam = 1e-2,
                 init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
                 mom_ts = 9.5, b_mom_ts = 9.5, sigma_mom_ts= 1, sigma_b_mom_ts= 1,
                 expected_wd=5e-4, max_wd=1e-3, prob_max_wd=1e-3)
    
    best_params, trials = tune_parameters(model=model,
                                          it_specs= optimizer.it_specs.get_type(),
                                          mode = optimizer.prox.mode,
                                          space = space,
                                          params_output_file = params_output_file,
                                          trials_output_file = trials_output_file,
                                          data = 'mnist',
                                          transform_train = transform_train,
                                          train_batch_size=128,
                                          subset_Data= 2**13,
                                          k_folds=1,
                                          num_epoch=10,
                                          max_evals=100)
    
    best_params = torch.load(params_output_file)
    trials = torch.load(trials_output_file)
    
    trials_df = trials_to_df(trials, trials_df_output_file)
    
    model = mnist_resnet34().to(device)
    
    model_output_file = 'results/model_data/resnet/'+save_str+'_sparse_model_bayes_params_small_data.dat'
    progress_data_output_file = 'results/progress_data/resnet/'+save_str+'_sparse_training_progress_bayes_params_small_data.csv'
    
    # model_output_file = None
    # progress_data_output_file = None
    
    
    training_specs = IterationSpecs(
        step_size = 2 - init_params['init_lr'], 
        mom_ts = init_params['mom_ts'], 
        b_mom_ts = init_params['b_mom_ts'], 
        weight_decay = init_params['weight_decay'], 
        av_param = init_params['av_param'])
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=best_params['lam'], maximum_factor=500, mode = mode))
    

    
    progress_df = progress_dataframe(model=model,
                                     params=init_params,
                                     model_output_file=model_output_file,
                                     progress_data_output_file = progress_data_output_file,
                                     data = 'mnist',
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     optimizer = optimizer,
                                     training_specs = training_specs,
                                     train_batch_size=128,
                                     subset_Data= subset_Data,
                                     num_epoch= num_epoch,
                                     # At about every 3 epochs, halve step size and double averaging.
                                     epoch_updates= [6,9,12.15,18,21,24,27,30])
    
    print(time.time() - t0)

