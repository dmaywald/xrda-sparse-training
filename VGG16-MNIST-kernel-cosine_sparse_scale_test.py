

import time
import numpy as np
import math

import torch
import torchvision.transforms as transforms

from models import mnist_vgg16_bn
from training_algorithms import xRDA
from regularization import l1_prox
from training_algorithms import CosineSpecs
from utils import progress_dataframe
from utils import trials_to_df

from hyperparameter_optimization import tune_parameters
from hyperparameter_optimization import VggParamSpace

if __name__ == '__main__':
    t0 = time.time()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = mnist_vgg16_bn(num_classes=10).to(device)
    
    train_batch_size = 128
    num_epoch = 30
    subset_Data = None
    mode = 'kernel' # normal, channel, or kernel
    save_str = 'mnist_vgg16_kernel_cosine_specs'
    
    
    
    if subset_Data is not None:
        max_iter = math.ceil(subset_Data/train_batch_size) * num_epoch
        
    if subset_Data is None:
        max_iter = math.ceil(60000/train_batch_size) * num_epoch # 60000 is length of mnist data
    
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
          # transforms.RandomHorizontalFlip(), # I don't think a horizontal flip is appropriate for the MNIST dataset
          transforms.ToTensor()])
    
    transform_val = transforms.Compose(
        [transforms.Pad(2), # Padding to a 32 x 32 image so the output dimensions after convolution fits 
         transforms.ToTensor()])
    
    
    


    # for C in [1/8, 1/4, 1/2, 1, 2, 4, 8]:
    for C in [1/4, 1/2, 1, 2, 4, 8]:    
        params_output_file = 'results/bayes_opt_params/vgg/'+mode+'/'+'CosineSpecs'+'/sparse_scale_'+str(C)+'_'+save_str+'_bayes_params.dat'
        trials_output_file = 'results/bayes_opt_params/vgg/'+mode+'/'+'CosineSpecs'+'/sparse_scale_'+str(C)+'_'+save_str+'_bayes_trials.dat'
        trials_df_output_file = 'results/bayes_opt_params/vgg/'+mode+'/'+'CosineSpecs'+'/sparse_scale_'+str(C)+'_'+save_str+'_bayes_trials.csv'
        
        # params_output_file = None
        # trials_output_file = None
        # trials_df_output_file  = None
        
        model = mnist_vgg16_bn(num_classes=10).to(device)
        
        # experimentally determined 5e-4 is too large for lambda
        space = VggParamSpace(expected_lam = 1e-6, max_lam = 5e-4, prob_max_lam = 1e-2,
                     init_lr_low = 0, init_lr_high = np.log(2), av_low = 0, av_high = 1,
                     mom_ts = 9.5, b_mom_ts = 9.5, sigma_mom_ts= 1, sigma_b_mom_ts= 1,
                     expected_wd= None)
        
        best_params, trials = tune_parameters(model=model,
                                              it_specs= 'CosineSpecs',
                                              mode = mode,
                                              space = space,
                                              params_output_file = params_output_file,
                                              trials_output_file = trials_output_file,
                                              data = 'mnist',
                                              transform_train = transform_train,
                                              train_batch_size=128,
                                              subset_Data= None,
                                              k_folds=1,
                                              num_epoch=2,
                                              max_evals= 40,
                                              max_iter= max_iter, 
                                              sparse_scale = C)
        
        best_params = torch.load(params_output_file)
        trials = torch.load(trials_output_file)
        trials_df = trials_to_df(trials, trials_df_output_file)
        
        model = mnist_vgg16_bn(num_classes=10).to(device)
        
        model_output_file = 'results/model_data/vgg/'+mode+'/'+'CosineSpecs'+'/sparse_scale_'+str(C)+'_'+save_str+'_bayes_params.dat'
        progress_data_output_file = 'results/progress_data/vgg/'+mode+'/'+'CosineSpecs'+'/sparse_scale_'+str(C)+'_'+save_str+'_bayes_params.csv'
        onnx_output_file = 'onnx_results/vgg/'+mode+'/'+'CosineSpecs'+'/sparse_scale_'+str(C)+'_'+save_str+'_bayes_params.onnx'
        
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
                                         data = 'mnist',
                                         transform_train = transform_train,
                                         transform_val = transform_val,
                                         optimizer = optimizer,
                                         training_specs = training_specs,
                                         train_batch_size=128,
                                         subset_Data=subset_Data,
                                         num_epoch=num_epoch,
                                         onnx_output_file= onnx_output_file)
    
    print(time.time() - t0)


