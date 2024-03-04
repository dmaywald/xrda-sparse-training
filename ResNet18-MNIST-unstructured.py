
import time
import numpy as np

import torch
import torchvision.transforms as transforms

from models import mnist_resnet18
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import IterationSpecs
from utils import progress_dataframe

from hyperparameter_optimization import tune_parameters
from hyperparameter_optimization import ResNetParamSpace


if __name__ == '__main__':
    t0 = time.time()

    init_params = {
        'init_lr': 1.0, #1.0 by default
        'lam': 1.3e-6, #1.3e-6 by default
        'av_param': 0.0, # 0.0 by default
        'mom_ts': 9.5, # 9.5 by default
        'b_mom_ts': 9.5, # 9.5 by default
        'weight_decay': 5e-4 # 5e-4 by default
        }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = mnist_resnet18(num_classes = 10).to(device)
    
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
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500))
    
    
    # model_output_file = 'results/model_data/resnet/mnist_resnet18_sparse_model_init_params_.dat'
    # progress_data_output_file = 'results/progress_data/resnet/mnist_resnet18_sparse_training_progress_init_params_.csv'
    
    model_output_file = None
    progress_data_output_file = None
    
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
                                     subset_Data=2048,
                                     num_epoch=2,
                                     # At about every 4 epochs, halve step size and double averaging.
                                     epoch_updates=[6, 10, 14, 18, 22, 26, 30, 34, 38, 42])
    
    # From the dataframe above:
    #     Training accuracy reaches maximum 99% after 5 epochs on half of the dataset
    #     Testing accuracy reaches 99% after 8 epochs and 99.5% after 16 epochs
    #     Sparsity is 77% after 10 epochs and 86% after 20 epochs
    #     Cross Entropy Loss is minimized to [.003, .15] after 5 epochs
    # So 10 epochs on the full data set should be used for 'num_epoch' in bayesian optimization

    params_output_file = 'results/bayes_opt_params/resnet/mnist_resnet18_sparse_model_bayes_params_small_data.dat'
    trials_output_file = 'results/bayes_opt_params/resnet/mnist_resnet18_sparse_model_bayes_trials_small_data.dat'
    
    # params_output_file = None
    # trials_output_file = None
    
    space = ResNetParamSpace(expected_lam = 1.3e-6, prob_max_lam = 5e-3, prob_max = .01,
                 init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
                 mom_ts = 9.5, b_mom_ts = 9.5)
    
    best_params, trials = tune_parameters(model=model,
                                          model_type = 'resnet',
                                          mode = 'normal',
                                          space = space,
                                          params_output_file = params_output_file,
                                          trials_output_file = trials_output_file,
                                          data = 'mnist',
                                          transform_train = transform_train,
                                          train_batch_size=128,
                                          subset_Data= 2**13,
                                          k_folds=1,
                                          num_epoch=5,
                                          max_evals=100)
    
    best_params = torch.load(params_output_file)
    trials = torch.load(trials_output_file)
    
    model_output_file = 'results/model_data/resnet/mnist_resnet18_sparse_model_bayes_params_small_data.dat'
    progress_data_output_file = 'results/progress_data/resnet/mnist_resnet18_sparse_training_progress_bayes_params_small_data.csv'
    
    # model_output_file = None
    # progress_data_output_file = None
    
    
    training_specs = IterationSpecs(
        step_size = 2 - best_params['init_lr'], 
        mom_ts = best_params['mom_ts'], 
        b_mom_ts = best_params['b_mom_ts'], 
        weight_decay = best_params['weight_decay'], 
        av_param = best_params['av_param'])
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500))
    
    progress_df = progress_dataframe(model=model,
                                     params=best_params,
                                     model_output_file=model_output_file,
                                     progress_data_output_file = progress_data_output_file,
                                     data = 'mnist',
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     optimizer = optimizer,
                                     training_specs = training_specs,
                                     train_batch_size=128,
                                     subset_Data=None,
                                     num_epoch=50,
                                     # At about every 4 epochs, halve step size and double averaging.
                                     epoch_updates=[6, 10, 14, 18, 22, 26, 30, 34, 38, 42])
    
    print(time.time() - t0)
