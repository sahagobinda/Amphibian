import importlib
import datetime
import argparse
import time
import os, os.path
import pdb
from tqdm import tqdm

import torch
from torch.autograd import Variable

import parser as file_parser
from metrics.metrics import confusion_matrix
from utils import misc_utils
from main_multi_task import life_experience_iid, eval_iid_tasks

import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 
import copy 

model_root = os.path.expanduser('save_stats')

def imshow(img):
  # img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  plt.imshow(np.transpose(npimg, (1, 2, 0))) 
  plt.show()

def eval_class_tasks(model, seq_tasks_loader, args):

    model.eval()
    result = []
    for idx, tasks in enumerate(seq_tasks_loader):

        for t_id, task_loader in enumerate(tasks):
            rt = 0
            t = t_id + idx*args.num_datasets 
            for (i, (x, y)) in enumerate(task_loader):
                if args.cuda:
                    x = x.cuda()
                    y = y+idx*args.class_per_dataset
                _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
                rt += (p == y).float().sum()

            result.append(rt / len(task_loader.dataset))
    return result

def eval_single_tasks(model, seq_tasks_loader, args,task_id):

    model.eval()
    result = 0
    for idx, tasks in enumerate(seq_tasks_loader):
        for t_id, task_loader in enumerate(tasks):
            t = t_id + idx*args.num_datasets 
            if task_id == t:
                rt = 0
                for (i, (x, y)) in enumerate(task_loader):
                    if args.cuda:
                        x = x.cuda()
                        y = y+idx*args.class_per_dataset
                    _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
                    rt += (p == y).float().sum()

                result = rt / len(task_loader.dataset)
    return result.item()

def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []

    result_val_t = []
    result_test_t = []

    time_start = time.time()
    test_tasks = []
    val_tasks=[]

    task_test_result_LCA = []
    fs_5shot_record = []
    save_lr_list =[]

    for dataset_name, loader in inc_loader:
        test_tasks.append(loader.get_tasks("test"))
        val_tasks.append(loader.get_tasks("val")) 
    
    single_task_evaluator = eval_single_tasks
    evaluator = eval_class_tasks

    for idx, (dataset_name, loader) in enumerate(inc_loader):
        print ('>> Current Dataset: {} <<'.format(dataset_name))
        for t_id in range(loader.n_tasks):
            task_info, train_loader, _, _ = loader.new_task()
            task_info_5shot, train_loader_5shot = loader.new_task_5shot()
            # print (task_info)

            task_i = t_id + idx*args.num_datasets
            task_info["task"] = task_i
            task_info_5shot["task"] = task_i

            ## copy the old task parameters - Only needed in AMPHIBIAN-lambda varient 
            if task_i>0:
                model.old_net_params = []
                for params in model.net.parameters():
                    temp_param  = params.detach().clone()
                    temp_param.requires_grad = False
                    model.old_net_params.append(temp_param.cuda()) 

            task_i_test_result =[]
            for ep in range(args.n_epochs):
                model.real_epoch = ep

                # ## Few-shot -------------------------------------------------------------------------------------------
                # for (i, (x, y)) in enumerate(train_loader_5shot):
                    
                #     v_x = x
                #     v_y = y + idx*args.class_per_dataset
                    
                #     if args.arch == 'linear':
                #         v_x = x.view(x.size(0), -1)
                #     if args.cuda:
                #         v_x = v_x.cuda()
                #         v_y = v_y.cuda()

                #     model.train()
                #     train_acc_fs, test_acc_fs = model.observe_fs(Variable(v_x), Variable(v_y), task_info_5shot["task"], test_tasks, True)
                #     print('-'*100)
                #     print ('FS-Stats :: Task: {}, FS-Training Data: {}, Train Accuracy: {}, Test Accuracy: {}'.format(task_info_5shot["task"], 
                #                                             task_info_5shot["n_train_data"], round(train_acc_fs,3), round(test_acc_fs,3)))
                #     print('-'*100)
                #     fs_5shot_record.append((round(train_acc_fs,3), round(test_acc_fs,3)))

                ## Online Continual Learning -------------------------------------------------------------------------------   
                prog_bar = tqdm(train_loader)
                for (i, (x, y)) in enumerate(prog_bar):

                    if((i % args.log_every) == 0):
                        result_val_a.append(evaluator(model, val_tasks, args))
                        result_val_t.append(task_info["task"])
                    
                    # # record LCA 
                    # task_i_test_result.append(single_task_evaluator(model, test_tasks, args, task_i))
                    
                    v_x = x
                    v_y = y + idx*args.class_per_dataset 

                    if args.arch == 'linear':
                        v_x = x.view(x.size(0), -1)
                    if args.cuda:
                        v_x = v_x.cuda()
                        v_y = v_y.cuda()

                    model.train()
                    if v_x.shape[0] !=10:
                        continue
                    else:
                        loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])
 
                        prog_bar.set_description(
                            "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                                task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                                round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                            )
                        )

            ## Save stats 
            # save_lr_list.append(copy.deepcopy(model.net.alpha_lr)) # this is a parameter list 
            # save LCA 
            task_test_result_LCA.append(task_i_test_result)

            # result_val_a.append(evaluator(model, val_tasks, args))
            # result_val_t.append(task_info["task"])

            if args.calc_test_accuracy:
                result_test_a.append(evaluator(model, test_tasks, args))
                result_test_t.append(task_info["task"])

    ## End of all tasks ----------------------------------------------------------------------------------------------------------------

    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a[-1])/len(result_test_a[-1]), result_test_a[-1]))

    ## Save results 
    if args.save_stats: 
        print ('Saving Stats ...')
        result_dict ={}
        result_dict['LCA'] = task_test_result_LCA
        result_dict['5-shot'] = fs_5shot_record
        result_dict['scales'] = save_lr_list
        file_name = os.path.join(model_root, '{}_{}_{}.pth'.format('fivedata', args.seed, args.unique_id))
        torch.save (result_dict, file_name)

    time_end = time.time()
    time_spent = time_end - time_start
    return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent #, task_test_result_LCA

def save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time):
    fname = os.path.join(args.log_dir, 'results')

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(result_val_t, result_val_a, args.log_dir, 'results.txt')
    
    one_liner = str(vars(args)) + ' # val: '
    one_liner += ' '.join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'results.txt', test=True)
        one_liner += ' # test: ' +  ' '.join(["%.3f" % stat for stat in test_stats])

    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_val_t, result_val_a, model.state_dict(),
                val_stats, one_liner, args), fname + '.pt')
    return val_stats, test_stats

def main():
    parser = file_parser.get_parser()

    args = parser.parse_args()

    # initialize seeds
    misc_utils.init_seed(args.seed)

    print('-'*100)
    print('ID: {}, Seed: {}, Dataset_list: {}, Glances: {}, Scale-LR: {}, Lambda-Init: {}, Reg: {}'.format(args.unique_id, args.seed,
                args.dataset_list, args.glances, args.opt_lr, args.alpha_init, args.weight_reg))
    print('-'*100)

    n_inputs, n_outputs, n_tasks = 0, 0, 0 
    final_loader =[]

    ## seq loader 
    print(args.dataset_list)
    print ('Loading datasets ...')
    for dataset_name in args.dataset_list.split("-"):
        args.dataset = dataset_name
        print (dataset_name, args.dataset)
        Loader = importlib.import_module('dataloaders.' + args.loader)
        loader = Loader.IncrementalLoader(args, seed=args.seed)
        n_inputs_t, n_outputs_t, n_tasks_t = loader.get_dataset_info()
        n_inputs = max(n_inputs_t, n_inputs)
        n_outputs +=n_outputs_t
        n_tasks += n_tasks_t
        final_loader.append([dataset_name,loader])
        # print (n_inputs_t, n_outputs_t, n_tasks_t, n_inputs, n_outputs, n_tasks )
        # print ( n_inputs, n_outputs, n_tasks )
   
    # set dataset command as 'cifar100' for code framework conveniance only - cifar100 won't be used  
    args.dataset = "cifar100"
    args.n_tasks = n_tasks
    args.num_datasets = int(len(final_loader))
    args.class_per_dataset = int(n_outputs/args.num_datasets)
    print ('n_inputs: {}, n_outputs: {}, n_tasks: {}, num_datasets: {}, class_per_dataset: {}'.format(n_inputs, 
                                                            n_outputs, n_tasks, args.num_datasets, args.class_per_dataset )) 

    # ## Sanity check 
    # for dataset_name, loader in final_loader:
    #     print('-'*30)
    #     print(dataset_name)
    #     for task_i in range(loader.n_tasks):
    #         task_info, train_loader, _, _ = loader.new_task()
    #         print (task_info)
    #         task_info_5shot, train_loader_5shot = loader.new_task_5shot()
    #         print (task_info_5shot)
    #         sample_iter = iter(train_loader)
    #         x_i, y_i = next(sample_iter)                   
    #         print(x_i.max(), x_i.min(), y_i.max(), y_i.min(), x_i.shape)
    #         sample_iter = iter(train_loader_5shot)
    #         x_i, y_i = next(sample_iter)                   
    #         print(x_i.max(), x_i.min(), y_i.max(), y_i.min(), x_i.shape)
    #         # imshow(torchvision .utils.make_grid(x_i[0]))
    #     print('-'*30)
    #     pdb.set_trace()
    # setup logging
    timestamp = misc_utils.get_date_time()
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    # misc_utils.print_model_report(model)
    if args.cuda:
        try:
            model.net.cuda()            
        except:
            pass 

    # run model on loader
    if args.model == "iid2":
        # oracle baseline with all task data shown at same time
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience_iid(
                                                                            model, final_loader, args)
    else:
        # for all the CL baselines
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
                                                                            model, final_loader, args)

        # save results in files or print on terminal
        save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)

    print('-'*100)
    print('ID: {}, Seed: {}, Dataset_list: {}, Glances: {}, Scale-LR: {}, Lambda-Init: {}, Reg: {}'.format(args.unique_id, args.seed,
                args.dataset_list, args.glances, args.opt_lr, args.alpha_init, args.weight_reg))
    print('-'*100)

if __name__ == "__main__":
    main()
    exit()