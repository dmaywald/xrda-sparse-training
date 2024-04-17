import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Subset
from test_accuracy import test_accuracy
    
def progress_dataframe(model, params, model_output_file, progress_data_output_file, data,
                       transform_train, transform_val, optimizer, training_specs,
                       train_batch_size = 128, subset_Data = None, num_epoch = 50, epoch_updates = None,
                       onnx_output_file = None):
    """
    
    
    Runtime Warning:
        
        Runtime is O((min(subset_Data,data)/train_batch_size) * num_epochs)
        
        
    Parameters
    ----------
    model : nn.Module object
        The Neural Network to be trained 
    params : dict
        Dictionary of tuning parameter used to train model. Should contain:
            {
                'lam': float of regularization strength [0,infinity)
                'init_lr': intial step size on [1,2] interval. 2 - 'init_lr' will be used.
                'av_param': float of averaging parameter [0,1]
                ''
                }
    model_output_file : string
        path and and name that model will be saved as. Use None if model should not be saved
    progress_data_output_file : string
        path and name of progress dataframe that will be saved as csv. Use None if progress dataframe should not be saved
    data : string
        Specify which dataset is being used. 
            Options: 'mnist', 'cifar10', 'cifar100' 
    transform_train : torchvision.transform object
        transformation made to training data
    transform_val : torchvision.transform object
        transformation made to validation/testing data
    optimizer : torch.optim.optimizer/xRDA object.
        optimizer used to train model. Should be class object with training specs already specified
    training_specs : class
        class object tracking training specs of model
    train_batch_size : int, optional
        Batch size of training data. The default is 128.
    subset_Data : int, optional
        Size of random subset of training data. Set to None for no subsetting. The default is None.
    num_epoch : int, optional
        Number of epochs to train model. An epoch is defined as one cycle through the training data. The default is 50.
    epoch_updates : list of integers, optional
        List of integers specifying when the averaging parameter should increased and step size should be halved.
        Set to None if the these training specs should be constant throughout training. The default is None.
    onnx_output_file : str, optional
        path and name of onnx visualization file that will be saved. Use None if onnx data should not be saved
    Returns
    -------
    Returns progress data frame after training model with given tuning/hyper parameters.
    
    Progress dataframe keeps track of:
        Epoch,
        loss per step,
        Training Accuracy per step,
        Testing Accuracy per end of Epoch,
        Sparsity per step,
        step_size per step,
        mom_ts per step,
        b_mom_ts per step,
        weight_decay per step,
        av_param per step
        lamda per step
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # If model_output_file has subdirectories, make sure those subdirectories exist  
    if model_output_file is not None:
        str_list = model_output_file.split("/")[:-1]
        if len(str_list) > 0:  
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
                    
    # If progress_data_output_file has subdirectories, make sure those subdirectories exist            
    if progress_data_output_file is not None:
        str_list = progress_data_output_file.split("/")[:-1]
        if len(str_list) > 0: 
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)
                    
    # If onnx_output_file has subdirectories, make sure those subdirectories exist            
    if onnx_output_file is not None:
        str_list = onnx_output_file.split("/")[:-1]
        if len(str_list) > 0: 
            temp_str = ''
            for idx in range(len(str_list)):
                temp_str = temp_str + str_list[idx]+'/'
                if not os.path.exists(temp_str):
                    os.mkdir(temp_str)                    
    
    # if subset data is used, load full data and get data subset sizes
    if subset_Data is not None:
        
        if data == "mnist":
            train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform_train)
            val_dataset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform_val)
            
        if data == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_val)
            
        if data == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_val)
        
        # Define the desired subset size
        subset_train_size = subset_Data
        subset_val_size = int(np.ceil(len(val_dataset) / (len(train_dataset)/subset_train_size)))
    
        # Create a subset of the MNIST dataset for analysis purposes
        subset_train_indices = torch.randperm(len(train_dataset))[:subset_train_size]
        subset_val_indices = torch.randperm(len(val_dataset))[:subset_val_size]
        
        trainset = Subset(train_dataset, subset_train_indices)
        testset = Subset(val_dataset, subset_val_indices)    
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                      shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                      shuffle=False, num_workers=2)
        
        
    
    # if subset data is not used, load full data
    if subset_Data is None:

        if data == "mnist":
            train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform_train)
            val_dataset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform_val)
            
        if data == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_val)
            
        if data == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
            val_dataset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform_val)

        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                      shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(val_dataset, batch_size=128,
                                                      shuffle=False, num_workers=2)
        
        
      
    model.train()
    criterion = nn.CrossEntropyLoss()
 
    av_param = params['av_param']
    
    lr = 2.0 - params['init_lr'] 
    
    num_model_params = sum([x.numel() for x in list(model.parameters())])
    len_per_epoch = sum(1 for _ in trainloader.batch_sampler)
    
    # DataFrame to keep track of progress, will contain:
    # Epoch, loss per step, Training Accuracy per step, Testing Accuracy per end of Epoch,
    # Sparsity per step, step_size per step, mom_ts per step, b_mom_ts per step,
    # weight_decay per step, av_param per step,( and lambda/mode/maximum factor per step if available).
    

    df_len = num_epoch*len_per_epoch
    
    # if l1 prox is available, track lambda. Otherwise, do not track lambda
    if optimizer.has_prox: 
        df_col_names = ['Epoch', 'loss', 'Train_acc', 'Epoch_Final_Test_acc', 'Sparsity',
                        'step_size', 'mom_ts', 'b_mom_ts', 'weight_decay', 'av_param', 'lam', 'maximum_factor']
        
    if not optimizer.has_prox: 
        df_col_names = ['Epoch', 'loss', 'Train_acc', 'Epoch_Final_Test_acc', 'Sparsity',
                        'step_size', 'mom_ts', 'b_mom_ts', 'weight_decay', 'av_param']
    
    progress_df = pd.DataFrame(np.nan, index = [i for i in range(df_len)], columns= df_col_names)
    
    # Fix the epoch column of dataframe since it will not change through training
    for i in range(num_epoch):
        progress_df.Epoch[i*(len_per_epoch):(i+1)*len_per_epoch] = i+1
    
    progress_df_idx = 0
    
    for epoch in range(num_epoch):
        # For each epoch, if subset of data is used, make training/testing data a subset of full data
        # At about every 4 epochs, halve step size and double averaging.
        if epoch_updates is not None and optimizer.it_specs.get_type() == 'IterationSpecs':
            if epoch+1 in epoch_updates:
                lr /= 2
                training_specs.set_step_size(lr)
                av_param = 1.0 - (1.0 - av_param) / 2.0
                training_specs.set_av_param(av_param)
                
        if subset_Data is not None:
            # Create a subset of the dataset for possible analysis purposes
            subset_train_indices = torch.randperm(len(train_dataset))[:subset_train_size]
            subset_val_indices = torch.randperm(len(val_dataset))[:subset_val_size]
            
            trainset = Subset(train_dataset, subset_train_indices)
            testset = Subset(val_dataset, subset_val_indices)    
            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                          shuffle=True, num_workers=4)
            testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                          shuffle=False, num_workers=2)
            
        total = 0
        correct = 0
        epoch_prog = 0
        for data in trainloader:
            # get the inputs
            inputs, labels = data
            
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            
          # zero the parameter gradients
            optimizer.zero_grad()

          # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            progress_df.loss[progress_df_idx] = loss.item()
            
            loss.backward()
            optimizer.step()
            epoch_prog += 1
          # Calculate train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print('Epoch: ', epoch+1)
            print('Epoch progress: ', epoch_prog, '/', len_per_epoch, sep = '')
            correct += (predicted == labels).sum()
            progress_df.Train_acc[progress_df_idx] = 100*correct.item()/total
            sparsity= sum(torch.nonzero(x).size()[0] for x in list(model.parameters()))
            progress_df.Sparsity[progress_df_idx] = 1 - sparsity/num_model_params
            progress_df.step_size[progress_df_idx] = training_specs.step_size(optimizer.iteration)
            progress_df.mom_ts[progress_df_idx] = training_specs.mom_ts
            progress_df.b_mom_ts[progress_df_idx] = training_specs.b_mom_ts
            progress_df.weight_decay[progress_df_idx] = training_specs.wd
            
            if optimizer.has_prox: # If l1 prox is available, update progress df with lambda and mode
                progress_df.lam[progress_df_idx] = optimizer.prox.lam
                progress_df.maximum_factor[progress_df_idx] = optimizer.prox.maximum_factor
            
            try: # if training specs has av attribute, update progress_df
                av_update = training_specs.av
            except AttributeError:
                av_update = training_specs.av_param(optimizer.iteration)
            
            progress_df.av_param[progress_df_idx] = av_update
            print('Current Loss: ', loss.item())
            print('Current Train Accuracy:', 100*correct.item()/total)
            print('Current Sparsity:', 100*(1 - sparsity/num_model_params))
            
            progress_df_idx += 1
            print('')
            
        
        accuracy = 10000 * correct / total
        sparsity = 10000 * (1- sparsity/num_model_params)
        if device.type == "cpu":
            t_accuracy = test_accuracy(testloader, model, cuda=False)
        else:
            t_accuracy = test_accuracy(testloader, model, cuda=True)
        
        progress_df.Epoch_Final_Test_acc[(epoch)*len_per_epoch:(epoch+1)*len_per_epoch] = t_accuracy.item()/100
        print('Epoch:%d %% Training Accuracy: %d.%02d %% Test Accuracy: %d.%02d %% \nSparsity Percentage: %d.%02d %%' % (epoch+1,
                                accuracy / 100, accuracy % 100, t_accuracy / 100, t_accuracy % 100, sparsity/100, sparsity % 100))
        
        print('')


    # Calculate accuracy and save output.
    if device.type == "cpu":
      final_accuracy = test_accuracy(testloader, model, cuda=False)
    else:
      final_accuracy = test_accuracy(testloader, model, cuda=True)
      
    print('Accuracy of the network on the %d test images: %d.%02d %%' %
          (len(val_dataset), final_accuracy / 100, final_accuracy % 100))
    
    if progress_data_output_file is not None:
        progress_df.to_csv(progress_data_output_file, index = True)

    if model_output_file is not None:
        checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, model_output_file)
        
    if onnx_output_file is not None:
        torch.onnx.export(model, inputs, onnx_output_file)
    

    return progress_df