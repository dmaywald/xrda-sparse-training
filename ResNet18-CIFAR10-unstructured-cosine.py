import time
import numpy as np
import math

import torch
import torchvision.transforms as transforms

from models import resnet18
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import CosineSpecs
from utils import progress_dataframe
from utils import trials_to_df

from hyperparameter_optimization import tune_parameters
from hyperparameter_optimization import ResNetParamSpace


if __name__ == '__main__':
    t0 = time.time()

    init_params = {
        'init_lr': 1.0, #1.0 by default
        'lam': 1e-7, #1e-7 by default
        'av_param': 0.0, # 0.0 by default
        'mom_ts': 9.5, # 9.5 by default
        'b_mom_ts': 9.5, # 9.5 by default
        'weight_decay': 5e-4 # 5e-4 by default
        }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = resnet18().to(device)
    
    train_batch_size = 128
    num_epoch = 40
    subset_Data = None
    mode = 'normal' # normal, channel, or kernel
    save_str = 'cifar10_resnet18_unstructured_cosine_specs'
    data = 'cifar10'
    
    if subset_Data is not None:
        max_iter = math.ceil(subset_Data/train_batch_size) * num_epoch
        
    if subset_Data is None:
        max_iter = math.ceil(50000/train_batch_size) * num_epoch # 50000 is length of CIFAR10 data
    
        
    transform_train = transforms.Compose(
    
          [transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))])
    
    transform_val = transforms.Compose(
    
          [transforms.ToTensor(),
           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))])
        
    
    training_specs = CosineSpecs(max_iter=max_iter, 
                                 init_step_size= 2 - init_params['init_lr'],
                                 mom_ts=init_params['mom_ts'],
                                 b_mom_ts=init_params['b_mom_ts'],
                                 weight_decay=init_params['weight_decay'])
    
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=init_params['lam'], maximum_factor=500, mode= mode))
    
    
    model_output_file = 'results/model_data/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_init_params.dat'
    progress_data_output_file = 'results/progress_data/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_init_params.csv'
    onnx_output_file = 'onnx_results/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_init_params.onnx'
    
    # model_output_file = None
    # progress_data_output_file = None
    # onnx_output_file = None
    
    progress_df = progress_dataframe(model=model,
                                     params=init_params,
                                     model_output_file=model_output_file,
                                     progress_data_output_file = progress_data_output_file,
                                     data = data,
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     optimizer = optimizer,
                                     training_specs = training_specs,
                                     train_batch_size=128,
                                     subset_Data=subset_Data,
                                     num_epoch=num_epoch,
                                     onnx_output_file= onnx_output_file)



    params_output_file = 'results/bayes_opt_params/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_bayes_params.dat'
    trials_output_file = 'results/bayes_opt_params/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_bayes_trials.dat'
    trials_df_output_file = 'results/bayes_opt_params/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_bayes_trials.csv'
    
    # params_output_file = None
    # trials_output_file = None
    # trials_df_output_file  = None
    
    model = resnet18().to(device)
    
    

    space = ResNetParamSpace(expected_lam = 1e-7, max_lam = 2.5e-6, prob_max_lam = 1e-2,
                 init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
                 mom_ts = 9.5, b_mom_ts = 9.5, sigma_mom_ts= 1, sigma_b_mom_ts= 1,
                 expected_wd= None)
    
    best_params, trials = tune_parameters(model=model,
                                          it_specs= optimizer.it_specs.get_type(),
                                          mode = optimizer.prox.mode,
                                          space = space,
                                          params_output_file = params_output_file,
                                          trials_output_file = trials_output_file,
                                          data = data,
                                          transform_train = transform_train,
                                          train_batch_size=128,
                                          subset_Data= subset_Data,
                                          k_folds=1,
                                          num_epoch=15,
                                          max_evals= 10,
                                          max_iter= max_iter)
    
    best_params = torch.load(params_output_file)
    trials = torch.load(trials_output_file)
    trials_df = trials_to_df(trials, trials_df_output_file)
    
    model = resnet18().to(device)
    
    model_output_file = 'results/model_data/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_bayes_params.dat'
    progress_data_output_file = 'results/progress_data/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_bayes_params.csv'
    onnx_output_file = 'onnx_results/resnet/'+mode+'/'+optimizer.it_specs.get_type()+'/'+save_str+'_bayes_params.onnx'
    
    # model_output_file = None
    # progress_data_output_file = None
    # onnx_output_file = None
    
    
    training_specs = CosineSpecs(max_iter=max_iter, 
                                 init_step_size= 2 - best_params['init_lr'],
                                 mom_ts= best_params['mom_ts'],
                                 b_mom_ts= best_params['b_mom_ts'],
                                 weight_decay= best_params['weight_decay'])
    
    optimizer = xRDA(model.parameters(), it_specs=training_specs, 
                     prox=l1_prox(lam=best_params['lam'], maximum_factor=500, mode = mode))
    

    
    progress_df = progress_dataframe(model=model,
                                     params=best_params,
                                     model_output_file=model_output_file,
                                     progress_data_output_file = progress_data_output_file,
                                     data = data,
                                     transform_train = transform_train,
                                     transform_val = transform_val,
                                     optimizer = optimizer,
                                     training_specs = training_specs,
                                     train_batch_size=128,
                                     subset_Data=subset_Data,
                                     num_epoch=num_epoch,
                                     onnx_output_file= onnx_output_file)
    
    print(time.time() - t0)

