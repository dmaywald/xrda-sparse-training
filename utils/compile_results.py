import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sns.set(rc={"figure.dpi":300,
            'savefig.dpi':300,
            'figure.figsize': (8*1.1, 6*1.1),
            'font.size': 12.0})

sns.set_context('notebook')
sns.set_style("ticks")
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 

os.chdir(parent)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collect_csv_data(great_grand_par, grand_par, par):
    df_list = []
    results_list = []
    dir_name = 'results/progress_data/'+great_grand_par+'/'+grand_par+'/'+par+'/'
    bayes_params_dir = 'results/bayes_opt_params/'+great_grand_par+'/'+grand_par+'/'+par+'/'
    model_data_dir = 'results/model_data/'+great_grand_par+'/'+grand_par+'/'+par+'/'
    file_list = os.listdir(dir_name)
    # print(os.listdir(bayes_params_dir))
    # print(great_grand_par, grand_par, par)

    for file in file_list:
        string_list = file.split("_")
        if string_list[0] == 'sparse':
            sparse_scale = string_list[2]
            data = string_list[3]
            model = string_list[4]
            struct = string_list[5]
            train_specs = string_list[6]
            params_type = string_list[8]
            # get initial av_param & intial step_size from results/bayes_opt_params/.../.../.../ directory
            bayes_load_str = bayes_params_dir + "_".join(string_list[0:-1])+"_params.dat"
            bayes_params = torch.load(bayes_load_str)
            init_av = bayes_params['av_param']
            init_step_size = 2 - bayes_params['init_lr']
            # get model data to count number of sparse kernels/channels ?
            # model_data_load_str = model_data_dir + "_".join(string_list[0:-1])+"_params.dat"
            # model_data = torch.load(model_data_load_str,  map_location=device)
            
            
        else:
            data = string_list[0]
            model = string_list[1]
            struct = string_list[2]
            train_specs = string_list[3]
            params_type = string_list[5]
            if params_type == 'init':
                sparse_scale = '-'
                init_av = 0
                init_step_size = 1
                
                
            else:
                sparse_scale = '1'
                # get initial av_param & intial step_size from results/bayes_opt_params/.../.../.../ directory
                bayes_load_str = bayes_params_dir + "_".join(string_list[0:-1])+"_params.dat"
                bayes_params = torch.load(bayes_load_str)
                init_av = bayes_params['av_param']
                init_step_size = 2 - bayes_params['init_lr']
                
        df = pd.read_csv(dir_name+file, header = 0)
        df = df.assign(init_step_size = init_step_size)
        df = df.assign(init_av = init_av)
        df = df.assign(data = data)
        df = df.assign(model = model)
        df = df.assign(structure = struct)
        df = df.assign(train_sepcs = train_specs)    
        df = df.assign(params_type = params_type)
        df = df.assign(sparse_scale = sparse_scale)
        df_list.append(df)
        results_list.append(df.iloc[-1,:])

    
    final_content = pd.concat([result for result in results_list], axis = 1).T 
    float_cols = ["Unnamed: 0", "Epoch", "loss", "Train_acc", "Epoch_Final_Test_acc", "Sparsity",
     "mom_ts", "b_mom_ts", "weight_decay", "lam", "maximum_factor", 'init_step_size',
     "init_av"]
    final_content[float_cols] = final_content[float_cols].apply(pd.to_numeric)
    # ends = pd.Series(final_content['Unnamed: 0'] == max(final_content['Unnamed: 0']))
    results = final_content
    results.index = [i for i in range(len(results))]
    results = results.drop('step_size', axis = 1)
    results = results.drop('av_param', axis = 1)
    return results, df_list


results_list = []
final_content_list = []

# At least I kept everything consistent
great_grand_par_names = os.listdir('results/progress_data/')
for great_grand_par in great_grand_par_names:
    grand_par_names = os.listdir('results/progress_data/'+great_grand_par+'/')
    for grand_par in grand_par_names:
        par_names = os.listdir('results/progress_data/'+great_grand_par+'/'+grand_par+'/')
        for par in par_names:
            result_temp, final_content_temp = collect_csv_data(great_grand_par, grand_par, par)
            results_list.append(result_temp)
            final_content_list.append(final_content_temp)
        
final_result = pd.concat([result for result in results_list], axis = 0)

final_result.index = [i for i in range(len(final_result))]

final_result = final_result.assign(lam_rewrite = final_result['lam']*(10**6))
final_result = final_result.assign(a_0 = final_result['init_av'])
final_result = final_result.assign(s_0 = final_result['init_step_size'])
final_result = final_result.assign(top1 = final_result['Epoch_Final_Test_acc'])
final_result = final_result.assign(perc_non_zero = 1 - final_result["Sparsity"]) 
final_result = final_result.assign(compression_ration = 1/(1-final_result['Sparsity']))








# print(final_content) 
# plot_title = data.upper() + ": " + model.upper() + " " + train_specs.capitalize() + " Specs -- " + struct.capitalize() + ' Sparsity'
# ax = sns.lineplot(data = final_content,
#                   x = 'Unnamed: 0',
#                   y = "Sparsity",
#                   hue = 'model')
# ax.set_title(plot_title)
# ax.set(xlabel = "Step")
# ax.set_ylim(0,1)
# plt.show()


# ax = sns.lineplot(data = final_content,
#                   x = 'Unnamed: 0',
#                   y = "Train_acc",
#                   hue = 'model')
# ax.set_title(plot_title)
# ax.set(ylabel = "Training Accuracy")
# ax.set(xlabel = "Step")
# ax.set_ylim(final_content['Train_acc'].quantile(.01),100)
# plt.show()

# ax = sns.scatterplot(data = final_content,
#                   x = 'Epoch',
#                   y = "Epoch_Final_Test_acc",
#                   hue = 'model')
# ax.set_title(plot_title)
# ax.set(xlabel = "Epoch")
# ax.set(ylabel = "Testing Accuracy")
# ax.set_ylim(0,100)
# plt.show()

# ax = sns.scatterplot(data = final_content,
#                   x = 'Unnamed: 0',
#                   y = "loss",
#                   hue = 'model')
# ax.set_title("MNIST: VGG16 Cosine Specs -- Kernel Sparsity")
# ax.set(xlabel = "Epoch")
# ax.set(ylabel = "Cross Entropy Loss")
# plt.show()


