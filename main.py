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
import numpy as np 
import copy 

model_root = os.path.expanduser('save_stats')

def eval_class_tasks(model, tasks, args):
    model.eval()
    result = []
    for t, task_loader in enumerate(tasks):
        rt = 0

        for (i, (x, y)) in enumerate(task_loader):
            if args.cuda:
                x = x.cuda()
            _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
            rt += (p == y).float().sum()

        result.append(rt / len(task_loader.dataset))
    return result

def eval_single_tasks(model, tasks, args,task_id):
    model.eval()
    result = 0
    for t, task_loader in enumerate(tasks):
        if task_id == t:
            rt = 0

            for (i, (x, y)) in enumerate(task_loader):
                if args.cuda:
                    x = x.cuda()
                _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
                rt += (p == y).float().sum()

            result = rt / len(task_loader.dataset)
    return result.item()

def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []

    result_val_t = []
    result_test_t = []

    task_test_result_LCA = []
    fs_5shot_record = []
    save_lr_list =[]

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")
    
    single_task_evaluator = eval_single_tasks
    evaluator = eval_class_tasks
    
    for task_i in range(inc_loader.n_tasks):
        task_info, train_loader, _, _ = inc_loader.new_task()
        task_info_5shot, train_loader_5shot = inc_loader.new_task_5shot()
        # print (task_info)
        
        ## copy the old task parameters - Only needed in AMPHIBIAN-beta    
        if task_i>0:
            model.old_net_params = []
            for params in model.net.parameters():
                temp_param  = params.detach().clone()
                temp_param.requires_grad = False
                model.old_net_params.append(temp_param.cuda()) 

        task_i_test_result =[]
        for ep in range(args.n_epochs):
            model.real_epoch = ep

            # ## Few-shot Forward Transfer------------------------------------------------------------------------------
            # for (i, (x, y)) in enumerate(train_loader_5shot):
                
            #     v_x = x
            #     v_y = y
            #     if args.arch == 'linear':
            #         v_x = x.view(x.size(0), -1)
            #     if args.cuda:
            #         v_x = v_x.cuda()
            #         v_y = v_y.cuda()
            #     # pdb.set_trace()
            #     model.train()

            #     train_acc_fs, test_acc_fs = model.observe_fs(Variable(v_x), Variable(v_y), task_info_5shot["task"], test_tasks)
            #     print('-'*100)
            #     print ('FS-Stats :: Task: {}, FS-Training Data: {}, Train Accuracy: {}, Test Accuracy: {}'.format(task_info_5shot["task"], 
            #                                                 task_info_5shot["n_train_data"], round(train_acc_fs,3), round(test_acc_fs,3)))
            #     print('-'*100)
            #     fs_5shot_record.append((round(train_acc_fs,3), round(test_acc_fs,3)))

            ## Online Continual Learning ------------------------------------------------------------------------------- 
            prog_bar = tqdm(train_loader)
            for (i, (x, y)) in enumerate(prog_bar):

                if((i % args.log_every) == 0):
                    result_val_a.append(evaluator(model, val_tasks, args))
                    result_val_t.append(task_info["task"])
                
                # ## record LCA - for TLE calculation 
                # model.eval()
                # task_i_test_result.append(single_task_evaluator(model, test_tasks, args, task_i))
                
                v_x = x
                v_y = y
                if args.arch == 'linear':
                    v_x = x.view(x.size(0), -1)
                if args.cuda:
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()

                model.train()
                loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])

                prog_bar.set_description(
                    "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                        task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                        round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                    )
                )

        # Save stats 
        save_lr_list.append(copy.deepcopy(model.net.alpha_lr)) # this is a parameter list 
        
        # save LCA 
        task_test_result_LCA.append(task_i_test_result)

        # result_val_a.append(evaluator(model, val_tasks, args))
        # result_val_t.append(task_info["task"])

        if args.calc_test_accuracy:
            result_test_a.append(evaluator(model, test_tasks, args))
            result_test_t.append(task_info["task"])


    # pdb.set_trace()
    # print(task_test_result_LCA)
    # print(args.num_fs_updates, args.alpha_init)
    # print(fs_5shot_record)

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
        file_name = os.path.join(model_root, '{}_{}_{}.pth'.format(args.dataset, args.seed, args.unique_id))
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
    print('ID: {}, Seed: {}, Dataset: {}, Glances: {}, Scale-LR: {}, Lambda-Init: {}, Reg: {}'.format(args.unique_id, args.seed,
                args.dataset, args.glances, args.opt_lr, args.alpha_init, args.weight_reg))
    print('-'*100)

    # set up loader
    # 2 options: class_incremental and task_incremental (this one is selected)
    Loader = importlib.import_module('dataloaders.' + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    print (n_inputs, n_outputs, n_tasks)

    ## sanity checks for dataloader 
    # for task_i in range(loader.n_tasks):
    #     task_info, train_loader, _, _ = loader.new_task()
    #     # task_info_5shot, train_loader_5shot = loader.new_task_5shot()
    #     sample_iter = iter(train_loader)
    #     x_i, y_i = next(sample_iter)        
    #     print (task_info)
    #     print(x_i.max(), x_i.min(), y_i.max(), y_i.min())
    # pdb.set_trace()

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
                                                                                    model, loader, args)
    else:
        # for all the CL baselines
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
                                                                                    model, loader, args)

        # save results in files or print on terminal
        save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)

    print('-'*100)
    print('ID: {}, Seed: {}, Dataset: {}, Glances: {}, Scale-LR: {}, Lambda-Init: {}, Reg: {}'.format(args.unique_id, args.seed,
                args.dataset, args.glances, args.opt_lr, args.alpha_init, args.weight_reg))
    print('-'*100)


if __name__ == "__main__":
    main()
    exit()